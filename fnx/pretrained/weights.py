import numpy as np
import tqdm, requests, os, sys, types, re, jax.tree_util, functools, fnx
import jax.numpy as jnp
from collections import defaultdict
import haiku as hk
from functools import partial

LOAD_PREFIX = "load_prefix"

def reshape(x, shape, name):
    if np.all(np.asarray(np.squeeze(x).shape) == np.asarray([s for s in shape if s != 1])):
        return np.reshape(x, shape)
    else:
        raise ValueError(f"Haiku value {name} expected weight shape {shape}, but got shape {x.shape}")

def common_prefix(strings):
    strings = list(strings)
    s = ""
    while all(len(s) < len(sn) for sn in strings) and len(set(sn[len(s)] for sn in strings)) == 1:
        s = s + strings[0][len(s)]
    return s

def common_postfix(strings):
    strings = list(strings)
    s = ""
    while all(len(s) < len(sn) for sn in strings) and len(set(sn[len(sn) - len(s) - 1] for sn in strings)) == 1:
        s = strings[0][len(strings[0]) - len(s) - 1] + s
    return s

def extract_number_expression(name):
    expr = ""
    is_num = False
    nums = 0
    for c in name:
        if c.isnumeric():
            is_num = True
        else:
            if is_num:
                is_num = False
                expr += "([0-9]+)"
                nums += 1
            expr += re.escape(c)
    if is_num:
        expr += "([0-9]+)"
        nums += 1
    expr = "^" + expr + "$"
    return expr, nums


class Node:
    def __init__(self, value, relative_prefix, full_prefix, direct_children, depth):
        self.value = value
        self.relative_prefix = relative_prefix
        self.full_prefix = full_prefix
        self.direct_children = direct_children
        self.parent = None
        self.depth = depth
        self.other = None
        if self.is_leaf():
            assert not value is None

    def is_leaf(self):
        return len(self.direct_children) == 0

    def is_predecessor_of(self, other):
        return id(other) == id(self) or (not other.parent is None and self.is_predecessor_of(other.parent))

    def get_structured_shapes(self, ignore_shape_one=True, remove_trivial_nodes=False, flatten=False):
        value_shape = [] if self.value is None else [tuple(s for s in self.value.shape if not ignore_shape_one or s != 1)]
        direct_children = self.direct_children
        if remove_trivial_nodes:
            def drop(n):
                if len(n.direct_children) == 1:
                    return drop(n.direct_children[0])
                else:
                    return n
            direct_children = [drop(c) for c in direct_children]
        children_shapes = [c.get_structured_shapes(ignore_shape_one=ignore_shape_one, remove_trivial_nodes=remove_trivial_nodes, flatten=flatten) for c in direct_children]
        if flatten:
            def flatten(shapes):
                result = []
                for s in shapes:
                    if isinstance(s, tuple) and all(not isinstance(x, tuple) for x in s):
                        result.append(s)
                    else:
                        result.extend(flatten(s))
                return result
            children_shapes = flatten(children_shapes)
        def compare_int(i1, i2):
            if i1 < i2:
                return -1
            elif i1 > i2:
                return 1
            else:
                return 0
        def compare(shapes1, shapes2):
            if isinstance(shapes1, tuple) and isinstance(shapes2, tuple):
                c = compare_int(len(shapes1), len(shapes2))
                if c != 0:
                    return c
                for s1, s2 in zip(shapes1, shapes2):
                    c = compare(s1, s2)
                    if c != 0:
                        return c
                return 0
            elif isinstance(shapes1, int) and isinstance(shapes2, tuple):
                return 1
            elif isinstance(shapes1, tuple) and isinstance(shapes2, int):
                return -1
            else:
                return compare_int(shapes1, shapes2)
        return tuple(sorted(value_shape + children_shapes, key=functools.cmp_to_key(compare)))

    def get_all_nodes(self):
        result = [self]
        for c in self.direct_children:
            result.extend(c.get_all_nodes())
        return result

    def printout(self, indentation=""):
        print(indentation, self.relative_prefix)
        if not self.value is None:
            print(indentation + "    " + self.value.name + " " + str(self.value.shape))
        for c in self.direct_children:
            c.printout(indentation + "    ")

def build_tree(values, value_names_without_prefix=None, parent_prefix="", depth=0):
    if value_names_without_prefix is None:
        value_names_without_prefix = [v.name for v in values]
    assert len(values) == len(value_names_without_prefix)
    assert len(values) > 0
    values = [v for v in values]
    value_names_without_prefix = [v for v in value_names_without_prefix]

    prefix = common_prefix(value_names_without_prefix)

    node_value = None
    children = {}
    for value, value_name_without_prefix in zip(values, value_names_without_prefix):
        if len(value_name_without_prefix) == len(prefix):
            assert node_value is None
            node_value = value
        else:
            c = value_name_without_prefix[len(prefix)]
            if not c in children:
                children[c] = ([], [])
            children[c][0].append(value)
            children[c][1].append(value_name_without_prefix[len(prefix):])
    children = [build_tree(values, value_names_without_prefix, parent_prefix=parent_prefix + prefix, depth=depth + 1) for values, value_names_without_prefix in children.values()]

    result = Node(
        value=node_value,
        relative_prefix=prefix,
        full_prefix=parent_prefix + prefix,
        direct_children=children,
        depth=depth,
    )
    for n in children:
        n.parent = result

    return result

def structured_shapes_depth(shape):
    if isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
        return 1
    else:
        assert all(not isinstance(s, int) for s in shape)
        return 1 + max(structured_shapes_depth(s) for s in shape)



def matcher(func):
    pairs_to_id = lambda pairs: sorted([tuple(set(n.full_prefix for n in nodes) for nodes in pair) for pair in pairs], key=lambda pair: len(pair[0]) + len(pair[1]))
    def wrapped(self, pairs_in, *args, **kwargs):
        for pair_in in pairs_in:
            assert isinstance(pair_in, tuple)
            assert len(pair_in) == 2
            assert isinstance(pair_in[0], list)
            assert isinstance(pair_in[1], list)

        all_pairs_out = []
        for pair_in in pairs_in:
            x = list(func(self, [n for n in pair_in[0]], [n for n in pair_in[1]], *args, **kwargs))
            pairs_out = [t[0] for t in x]
            descriptions_out = [t[1] for t in x]

            # All remaining nodes into single pair
            hk_rest = [n for n in pair_in[0]]
            in_rest = [n for n in pair_in[1]]
            assert len(hk_rest) == len(set(id(n) for n in hk_rest))
            assert len(in_rest) == len(set(id(n) for n in in_rest))
            for pair_out in pairs_out:
                assert len(pair_out[0]) == len(set(id(n) for n in pair_out[0]))
            hk_nodes_out = [hk_node for pair_out in pairs_out for hk_node in pair_out[0]]
            assert len(hk_nodes_out) == len(set(id(n) for n in hk_nodes_out))
            for pair_out in pairs_out:
                for hk_node in pair_out[0]:
                    hk_rest.remove(hk_node)
                for in_node in pair_out[1]:
                    in_rest.remove(in_node)
            pairs_out.append((hk_rest, in_rest))
            descriptions_out.append(None)

            # Remove empty or unmatchable pairs
            i = 0
            while i < len(pairs_out):
                if len(pairs_out[i][0]) == 0 or len(pairs_out[i][1]) == 0:
                    if not all(not n.is_leaf() for n in pairs_out[i][0]) or not all(not n.is_leaf() for n in pairs_out[i][1]):
                        def check(nodes, type):
                            if any(n.is_leaf() for n in nodes):
                                print(f"Matcher {func.__name__} yielded unmatched leafs:")
                                for n in nodes:
                                    if n.is_leaf():
                                        print(f"    {type} {n.full_prefix} {n.get_structured_shapes()}")
                        check(pairs_out[i][0], "HK")
                        check(pairs_out[i][1], "IN")
                        print()
                        print("Got HK values:")
                        for v in self.hk_values:
                            print(f"    HK {v.name} {v.shape}")
                        print()
                        print("Got IN values:")
                        for v in self.in_values:
                            print(f"    IN {v.name} {v.shape}")
                        raise ValueError("Matcher yielded unmatched leafs")
                    del pairs_out[i]
                    del descriptions_out[i]
                else:
                    i += 1

            for pair_out, description in zip(pairs_out, descriptions_out):
                if len(pair_out[0]) == 1 and len(pair_out[1]) == 1:
                    self.pair_node(pair_out[0][0], pair_out[1][0])
                elif len(pair_out[0]) > 0 or len(pair_out[1]) > 0:
                    all_pairs_out.append(pair_out)

                if self.verbose and not description is None and (pairs_to_id([pair_in]) != pairs_to_id(pairs_out)):
                    print(description)
                    for n in pair_out[0]:
                        print(f"    HK {n.full_prefix} {n.get_structured_shapes()}")
                    for n in pair_out[1]:
                        print(f"    IN {n.full_prefix} {n.get_structured_shapes()}")

        changed = pairs_to_id(pairs_in) != pairs_to_id(all_pairs_out)
        return all_pairs_out, changed
    wrapped.__name__ = func.__name__
    return wrapped

def hint_matcher(func):
    def wrapped(self, hk_nodes, in_nodes):
        hints, is_match, hint_description = func(self, hk_nodes, in_nodes)

        for hint in hints:
            # Build score matrix for all pairs of nodes
            matrix = np.zeros((len(hk_nodes), len(in_nodes)), dtype="int32")
            for hk_index in range(len(hk_nodes)):
                for in_index in range(len(in_nodes)):
                    matrix[hk_index, in_index] = 1 if is_match(hk_nodes[hk_index], in_nodes[in_index], hint) else 0
            s = np.sum(matrix)
            if s < len(hk_nodes) or s == len(hk_nodes) * len(in_nodes):
                continue

            # Find contiguous groups of matched nodes in matrix
            index_groups = set()
            checked_pairs = set()
            for hk_index in range(len(hk_nodes)):
                for in_index in range(len(in_nodes)):
                    if not (hk_index, in_index) in checked_pairs:
                        if matrix[hk_index, in_index] == 1:
                            hk_indices = set()
                            in_indices = set()
                            hk_indices.add(hk_index)
                            in_indices.add(in_index)
                            for hk_index2 in range(len(hk_nodes)):
                                if matrix[hk_index2, in_index] == 1:
                                    hk_indices.add(hk_index2)
                                    checked_pairs.add((hk_index2, in_index))
                            for in_index2 in range(len(in_nodes)):
                                if matrix[hk_index, in_index2] == 1:
                                    in_indices.add(in_index2)
                                    checked_pairs.add((hk_index, in_index2))

                            hk_indices = tuple(sorted(hk_indices))
                            in_indices = tuple(sorted(in_indices))
                            index_groups.add((hk_indices, in_indices))
                        checked_pairs.add((hk_index, in_index))

            # Merge all groups with unequal number of nodes
            merged_hk_indices = set()
            merged_in_indices = set()
            index_groups_out = set()
            for hk_indices, in_indices in index_groups:
                if len(hk_indices) == len(in_indices):
                    index_groups_out.add((hk_indices, in_indices))
                else:
                    merged_hk_indices.update(hk_indices)
                    merged_in_indices.update(in_indices)
            if len(merged_hk_indices) != len(merged_in_indices):
                continue
            if len(merged_hk_indices) > 0:
                hk_indices = tuple(sorted(merged_hk_indices))
                in_indices = tuple(sorted(merged_in_indices))
                index_groups_out.add((hk_indices, in_indices))
            index_groups = index_groups_out

            # End if more than one group was found
            if len(index_groups) > 1:
                remaining_hk_nodes = [n for n in hk_nodes]
                remaining_in_nodes = [n for n in in_nodes]
                for hk_indices, in_indices in index_groups:
                    hk_group = [hk_nodes[i] for i in hk_indices]
                    in_group = [in_nodes[i] for i in in_indices]
                    for n in hk_group:
                        assert n in remaining_hk_nodes
                        remaining_hk_nodes.remove(n)
                    for n in in_group:
                        assert n in remaining_in_nodes
                        remaining_in_nodes.remove(n)
                    yield (hk_group, in_group), f"Found subgroup resulting from {hint_description(hint)}"
                yield (remaining_hk_nodes, remaining_in_nodes), None
                return

        yield (hk_nodes, in_nodes), None
    wrapped.__name__ = func.__name__
    wrapped = matcher(wrapped)
    return wrapped

class WeightMatcher:
    def __init__(self, in_values, separator=None, ignore_hk=lambda n: False, hints=[], verbose=False):
        if separator is None:
            possible_separators = [".", "/"]
            has_separator = [any(s in name for name in in_values.keys()) for s in possible_separators]
            if np.count_nonzero(has_separator) == 1:
                separator = possible_separators[np.argmax(has_separator, axis=0)]
            else:
                raise ValueError("Could not implicitly determine separator in weights. Please provide separator")
        self.in_values = in_values
        self.in_separator = separator
        self.ignore_hk = lambda n: ignore_hk(n) or n.endswith("/counter") or n.endswith("/hidden")
        self.verbose = verbose
        self.hk_values = None
        self.hints = hints

        self.paired_nodes = []

    def pair_node(self, hk_node, in_node):
        assert hk_node.is_leaf() == in_node.is_leaf()

        assert hk_node.other is None and in_node.other is None
        hk_node.other = in_node
        in_node.other = hk_node

        if hk_node.is_leaf():
            hk_node.value.other = in_node.value
            in_node.value.other = hk_node.value

        self.paired_nodes.append((hk_node, in_node))

    @matcher
    def match_unique_leafs(self, hk_nodes, in_nodes):
        yield (hk_nodes, in_nodes), "Found unique pair"

    @hint_matcher
    def match_by_paired_prefixes(self, hk_nodes, in_nodes):
        def get_prefixes(name, separator, nodes):
            prefixes = []
            prefix = ""
            for t in name.split(separator):
                prefix = (prefix + separator + t) if len(prefix) > 0 else t
                prefixes.append(prefix)
            def check(prefix):
                starting_with = list(n.full_prefix.startswith(prefix) for n in nodes)
                return any(starting_with)
            prefixes = [p for p in prefixes if check(p)]
            return prefixes

        hints = set()
        for hk_node, in_node in self.paired_nodes:
            hk_prefixes = get_prefixes(hk_node.full_prefix, "/", hk_nodes)
            in_prefixes = get_prefixes(in_node.full_prefix, self.in_separator, in_nodes)
            for hk_prefix in hk_prefixes:
                for in_prefix in in_prefixes:
                    hints.add((hk_prefix, in_prefix))
        hints = sorted(hints, key=lambda t: -(len(t[0]) + len(t[1])))

        def is_match(hk_node, in_node, hint):
            return hk_node.full_prefix.startswith(hint[0]) == in_node.full_prefix.startswith(hint[1])

        return hints, is_match, lambda hint: f"paired prefix hint HK {hint[0]} IN {hint[1]}"

    @hint_matcher
    def match_by_paired_predecessors(self, hk_nodes, in_nodes):
        def get_predecessors(node, nodes):
            parents = [node] if any(node.is_predecessor_of(n) for n in nodes) else []
            if not node.parent is None:
                parents = get_predecessors(node.parent, nodes) + parents
            return parents

        hints = set()
        for hk_node, in_node in self.paired_nodes:
            hk_predecessors = get_predecessors(hk_node, hk_nodes)
            in_predecessors = get_predecessors(in_node, in_nodes)
            for hk_predecessor in hk_predecessors:
                for in_predecessor in in_predecessors:
                    hints.add((hk_predecessor, in_predecessor))
        hints = sorted(hints, key=lambda t: -(t[0].depth + t[1].depth))

        def is_match(hk_node, in_node, hint):
            return hint[0].is_predecessor_of(hk_node) == hint[1].is_predecessor_of(in_node)

        return hints, is_match, lambda hint: f"paired predecessor HK {hint[0].full_prefix} IN {hint[1].full_prefix}"

    @hint_matcher
    def match_passed_hints(self, hk_nodes, in_nodes):
        def is_match(hk_node, in_node, hint):
            return (hint[0] in hk_node.full_prefix) == (hint[1] in in_node.full_prefix)

        return self.hints, is_match, lambda hint: f"passed hints"

    @hint_matcher
    def match_equivalent_hardcoded_leafs(self, hk_nodes, in_nodes):
        hints = [
            [["weight", "scale", "gamma", "w"], ["bias", "offset", "beta", "b"], ["moving_mean", "running_mean", "~/mean_ema/average", "mean_ema/average", "mean/average"], ["moving_variance", "running_var", "~/var_ema/average", "var_ema/average", "var/average"]],
        ]

        hk_nodes = [n for n in hk_nodes if n.is_leaf()]
        in_nodes = [n for n in in_nodes if n.is_leaf()]

        def get_index(node, separator, hint):
            for i, equivalent_postfixes in enumerate(hint):
                if any(node.full_prefix.endswith(separator + postfix) for postfix in equivalent_postfixes):
                    return i
            return -1

        def is_valid_hint(hint):
            return all(get_index(n, "/", hint) >= 0 for n in hk_nodes) and all(get_index(n, self.in_separator, hint) >= 0 for n in in_nodes)

        hints = [h for h in hints if is_valid_hint(h)]

        def is_match(hk_node, in_node, hint):
            return get_index(hk_node, "/", hint) == get_index(in_node, self.in_separator, hint)

        return hints, is_match, lambda hint: f"equivalent hardcoded leafs"

    @matcher
    def match_by_paired_parents(self, hk_nodes, in_nodes):
        def get_first_paired_parent(node):
            if not node.other is None:
                return node
            elif node.parent is None:
                return None
            else:
                return get_first_paired_parent(node.parent)

        def to_dict(nodes):
            pairedparent_to_nodes = {}
            for n in nodes:
                parent = get_first_paired_parent(n)
                if not parent is None:
                    if not parent.full_prefix in pairedparent_to_nodes:
                        pairedparent_to_nodes[parent.full_prefix] = (parent, [])
                    pairedparent_to_nodes[parent.full_prefix][1].append(n)
            return pairedparent_to_nodes

        hk_pairedparent_to_nodes = to_dict(hk_nodes)
        in_pairedparent_to_nodes = to_dict(in_nodes)

        for hk_parent, hk_children in hk_pairedparent_to_nodes.values():
            in_parent = hk_parent.other
            in_children = in_pairedparent_to_nodes[in_parent.full_prefix][1]
            yield (hk_children, in_children), f"Found subtrees with matched parents HK {hk_parent.full_prefix} IN {in_parent.full_prefix}"

    @matcher
    def match_by_structured_shapes(self, hk_nodes, in_nodes, ignore_shape_one=True, remove_trivial_nodes=False, flatten=False):
        pairs_to_id = lambda nodes: [n.full_prefix for n in nodes]

        def to_uniquekey_dict(nodes):
            result = defaultdict(list)
            for n in nodes:
                if n.other is None:
                    result[n.get_structured_shapes(ignore_shape_one=ignore_shape_one, remove_trivial_nodes=remove_trivial_nodes, flatten=flatten)].append(n)
            return result
        hk_uniquekey_to_nodes = to_uniquekey_dict(hk_nodes)
        in_uniquekey_to_nodes = to_uniquekey_dict(in_nodes)

        intersection_keys = set(hk_uniquekey_to_nodes.keys()).intersection(in_uniquekey_to_nodes.keys())
        intersection_keys = sorted(intersection_keys, key=lambda shapes: -structured_shapes_depth(shapes))

        for key in intersection_keys:
            yield (hk_uniquekey_to_nodes[key], in_uniquekey_to_nodes[key]), f"Found matches with structured shapes {key}"

        for key in hk_uniquekey_to_nodes.keys():
            if key not in intersection_keys:
                yield (hk_uniquekey_to_nodes[key], []), None
        for key in in_uniquekey_to_nodes.keys():
            if key not in intersection_keys:
                yield ([], in_uniquekey_to_nodes[key]), None

    @matcher
    def match_by_paired_children(self, hk_nodes, in_nodes):
        for pair_hk_node, pair_in_node in self.paired_nodes:
            hk_matches = [n for n in hk_nodes if pair_hk_node.full_prefix.startswith(n.full_prefix)]
            in_matches = [n for n in in_nodes if pair_in_node.full_prefix.startswith(n.full_prefix)]
            if len(hk_matches) == 1 and len(in_matches) == 1:
                hk_node = hk_matches[0]
                in_node = in_matches[0]
                yield ([hk_node], [in_node]), f"Matched by paired children of HK {pair_hk_node.full_prefix} IN {pair_in_node.full_prefix}"
                hk_nodes.remove(hk_node)
                in_nodes.remove(in_node)

    @matcher
    def match_number_regex(self, hk_nodes, in_nodes):
        def make_groups(nodes):
            nodes = [n for n in nodes]
            groupswithmatchingregex = []
            while len(nodes) > 1:
                # Find possible prefixes
                prefix_expressions = set()
                for expr, _ in set(extract_number_expression(n.full_prefix) for n in nodes):
                    prefix = ""
                    for t in expr.split("([0-9]+)")[:-1]:
                        prefix = prefix + t + "([0-9]+)"
                        prefix_expressions.add(prefix)
                prefix_expressions = sorted(prefix_expressions, key=lambda expr: len(expr))

                for prefix_expr in prefix_expressions:
                    prefix_expr = re.compile(prefix_expr)
                    matches = [prefix_expr.match(n.full_prefix) for n in nodes]
                    if sum((1 if m else 0) for m in matches) > 1:
                        matching_nodes = [n for n, m in zip(nodes, matches) if m]
                        matches = [m for m in matches if m]

                        # Find the first varying number in the list of numbers per node
                        numbers = np.asarray([[int(g) for g in match.groups()] for match in matches]) # matches nums
                        unique_nums = np.asarray([len(set(numbers[:, num_index])) for num_index in range(numbers.shape[1])]) # nums
                        for used_num_index in range(len(unique_nums)):
                            if unique_nums[used_num_index] > 1:
                                break
                        else:
                            continue
                        numbers = numbers[:, used_num_index] # matches

                        groupswithmatchingregex.append([(n, m) for n, m in zip(numbers, matching_nodes)])
                        for n in matching_nodes:
                            nodes.remove(n)
                        break # Found a prefix that counts more than one node
                else:
                    break # No prefix matches

            for n in nodes:
                groupswithmatchingregex.append([(None, n)])

            num_to_groupswithmatchingregex = defaultdict(list)
            for g in groupswithmatchingregex:
                num_to_groupswithmatchingregex[len(g)].append(g)

            return num_to_groupswithmatchingregex

        # Divide nodes into groups that look like: prefix number postfix
        hk_num_to_groupswithmatchingregex = make_groups(hk_nodes)
        in_num_to_groupswithmatchingregex = make_groups(in_nodes)

        # Pair groups that have the same unique size
        unique_nums = set(hk_num_to_groupswithmatchingregex.keys()).intersection(set(in_num_to_groupswithmatchingregex.keys()))
        for k in unique_nums:
            hk_groups = hk_num_to_groupswithmatchingregex[k]
            in_groups = in_num_to_groupswithmatchingregex[k]
            if len(hk_groups) == 1 and len(in_groups) == 1:
                # Unique number of nodes -> divide by parsed number values
                def to_dict(group):
                    nums_to_nodes = defaultdict(list)
                    for num, node in group:
                        nums_to_nodes[num].append(node)
                    return nums_to_nodes
                hk_nums_to_nodes = to_dict(hk_groups[0])
                in_nums_to_nodes = to_dict(in_groups[0])
                if len(hk_nums_to_nodes) == len(in_nums_to_nodes):
                    # Sort by parsed number values
                    hk_sorted_groups = [nodes for num, nodes in sorted(hk_nums_to_nodes.items(), key=lambda t: t[0])]
                    in_sorted_groups = [nodes for num, nodes in sorted(in_nums_to_nodes.items(), key=lambda t: t[0])]
                    assert len(hk_sorted_groups) == len(in_sorted_groups)

                    # Match
                    for hk_group, in_group in zip(hk_sorted_groups, in_sorted_groups):
                        yield (hk_group, in_group), "Found counted modules"

    @matcher
    def match_known_paired_number_regex(self, hk_nodes, in_nodes):
        last_pairs_num_name = "match_known_paired_number_regex_last_pairs_num"
        if not last_pairs_num_name in vars(self):
            vars(self)[last_pairs_num_name] = 0
        hints_name = "match_known_paired_number_regex_hints"
        if not hints_name in vars(self):
            vars(self)[hints_name] = {}
        hints = vars(self)[hints_name]

        for hk_node, in_node in self.paired_nodes[vars(self)[last_pairs_num_name]:]:
            hk_expr, hk_nums = extract_number_expression(hk_node.full_prefix)
            in_expr, in_nums = extract_number_expression(in_node.full_prefix)
            if hk_nums > 0 and in_nums > 0:
                hint = (hk_expr, in_expr)
                if not hint in hints:
                    hints[hint] = (re.compile(hint[0]), re.compile(hint[1]))
        vars(self)[last_pairs_num_name] = len(self.paired_nodes)

        def match(name, expr):
            match = re.match(expr, name)
            if match:
                return tuple(int(g) for g in match.groups())
            else:
                return None

        for hint in hints.values():
            def get_matches(nodes, hint):
                matches = [(n, match(n.full_prefix, hint)) for n in nodes]
                matches = [(n, nums) for n, nums in matches if not nums is None]
                matches = sorted(matches, key=lambda t: t[1])
                nums = [num for _, nums in matches for num in nums]
                if len(nums) != len(set(nums)):
                    return None
                matches = [n for n, nums in matches]
                return matches
            hk_matching_nodes = get_matches(hk_nodes, hint[0])
            in_matching_nodes = get_matches(in_nodes, hint[1])
            if not hk_matching_nodes is None and not in_matching_nodes is None and len(hk_matching_nodes) == len(in_matching_nodes) and len(hk_matching_nodes) > 0:
                for hk_node, in_node in zip(hk_matching_nodes, in_matching_nodes):
                    yield ([hk_node], [in_node]), f"Found counted nodes from paired hint HK {hint[0].pattern[1:-1]} IN {hint[1].pattern[1:-1]}:"
                    hk_nodes.remove(hk_node)
                    in_nodes.remove(in_node)

    def update_fn(self, hk_values):
        hk_values = {full_name: (shape, dtype, is_param) for full_name, (shape, dtype, is_param) in hk_values.items() if not self.ignore_hk(full_name)}
        if len(hk_values) == 0:
            raise ValueError("Found no Haiku parameters or state (that are not ignored)")
        self.hk_values = [types.SimpleNamespace(
            name=LOAD_PREFIX + "/" + full_name,
            shape=tuple(s for s in shape),
            dtype=dtype,
            other=None,
        ) for full_name, (shape, dtype, is_param) in hk_values.items()]

        # Create list of loaded values
        self.in_values = [types.SimpleNamespace(
            name=name,
            value=in_value,
            shape=tuple(s for s in in_value.shape),
            dtype=in_value.dtype,
            other=None,
        ) for name, in_value in self.in_values.items()]

        # Build module trees
        hk_tree = build_tree(self.hk_values)
        in_tree = build_tree(self.in_values)

        # Run matching heuristics
        ops = [
            self.match_by_structured_shapes,
            self.match_by_paired_parents,
            self.match_by_paired_children,
            self.match_unique_leafs,
            self.match_number_regex,
            self.match_known_paired_number_regex,
            self.match_equivalent_hardcoded_leafs,
            self.match_passed_hints,
            self.match_by_paired_predecessors,
            self.match_by_paired_prefixes,
        ]
        import time
        times = defaultdict(lambda: 0.0)

        proposals = [(
            [n for n in hk_tree.get_all_nodes() if n.other is None],
            [n for n in in_tree.get_all_nodes() if n.other is None],
        )]

        changed = True
        while changed:
            changed = False
            for op in ops:
                if self.verbose:
                    print(f"OP: Trying {op.__name__}")
                start = time.time()
                proposals, changed = op(proposals)

                times[op.__name__] += time.time() - start
                if changed:
                    if self.verbose:
                        print(f"OP: Changed by {op.__name__}")
                    break

        if self.verbose:
            print("Times (sec) per operation:")
            for k, v in sorted(times.items(), key=lambda t: t[1]):
                print(f"    {k} {v}")

        if len(proposals) > 0:
            print()
            for hk_nodes, in_nodes in proposals:
                print("Failed to pair the following nodes")
                for n in hk_nodes:
                    print(f"    HK {n.full_prefix} {n.get_structured_shapes()}")
                for n in in_nodes:
                    print(f"    IN {n.full_prefix} {n.get_structured_shapes()}")
            raise ValueError("Failed to pair loaded values with haiku values")

        if self.verbose:
            print()
            print("Paired values:")
            mapping = {hk_node.full_prefix: in_node.full_prefix for hk_node, in_node in self.paired_nodes if hk_node.is_leaf() and in_node.is_leaf()}
            for hk_value in self.hk_values:
                print(f"    {hk_value.name} -> {mapping[hk_value.name]}")

        # Matching successful
        matched_hk_values = {v.name: v for v in self.hk_values}
        hk_values = {hk_name: reshape(matched_hk_values[LOAD_PREFIX + "/" + hk_name].other.value, shape, hk_name).astype(dtype) for hk_name, (shape, dtype, is_param) in hk_values.items() if not self.ignore_hk(hk_name)}

        return hk_values





def init(func, weights, separator=None, ignore=lambda n: False, hints=[], verbose=False):
    """Returns a copy of ``func`` where weights are loaded from the ``weights`` dictionary even when the latter has been created using a different deep learning framework or source code repository. Automatically matches the source and target weights based on name and shape heuristics.

    **This can handle** the following cases:

    1. Different naming scheme for modules and weights.
    2. Different module tree structure/ grouping of submodules.
    3. Different weight shapes with added or removed trivial axes of size ``1``. This allows for example matching weights created using (1) a pixel-wise convolution layer and (2) a linear layer.

    **This cannot handle** functionally different layers in source and target model. In this case, the ``weights`` dictionary has to be preprocessed by the user to align the weights to the layers in ``func``. For example:

    1. Predicting queries, keys and values in a self-attention block can be done (1) using three separate linear layers or (2) using a single linear layer and splitting the output along the channel dimension into three groups.

       * Preprocessing for (1) -> (2): The single weight tensor has to be split along the output channel dimension into three separate weight tensors.
       * Preprocessing for (2) -> (1): The three separate weight tensors have to be concatenated along the output channel dimension into a single weight tensor.

    2. The learnable positional embedding in vision transformers can be added (1) on spatial tokens and the learnable class token or (2) only on the spatial tokens.

       * Preprocessing for (1) -> (2): Compute a new class token as the sum of the original class token and the corresponding component of the positional embedding. Remove the class-component from the positional embedding.
       * Preprocessing for (2) -> (1): Keep the class token as is. Add a zero-vector as the corresponding component to the positional embedding.

    TODO: Link to pretrained examples using this method

    In case of ambiguous matches, the matcher requires additional hints via the ``hints`` argument.

    Example usage:

    ..  code-block:: python

        @fnx.module
        def model(x):
            ...

        # Load the weights from a PyTorch checkpoint into a dictionary {name: weight}
        weights = fnx.pretrained.weights.load_pytorch("/path/to/weights/file")
        # Match and load the weights into our model
        model = fnx.pretrained.weights.init(model, weights)

    Parameters:
        func: Function representing the model into which weights should be loaded.
        weights: Dictionary ``{name: weight}`` from which weights are loaded.
        separator: Character used to separate module names in ``weights``. If this is ``None``, tries to automatically determine the separator is used in ``weights``. Defaults to ``None``.
        ignore: Function ``fn: name -> bool`` which returns True for weights in ``func`` that should not be included in the weight matching process. Defaults to ``lambda name: False``.
        hints: A list of hints for when when the automatic matching fails due to ambiguous matches. Each hint is a pair of matching (sub-)names in the target and source weights. For example:

          ..  code-block:: python

              model = fnx.pretrained.weights.init(model, weights, hints=[
                  ("name-of-module/in-fnx/model", "other.name-in.source-weights"),
                  ("second-hint", "second.hint/abc"),
              ])

          The names can be any substring of the corresponding names in the ambiguous match that resolve the ambiguity. Defaults to ``[]``.
        verbose: ``True`` if the matcher should print the output of every matching step, ``False`` otherwise.

    Returns:
        A copy of ``func`` with weights loaded from ``weights``.
    """

    update_fn = lambda values: WeightMatcher(weights, separator=separator, ignore_hk=ignore, hints=hints, verbose=verbose).update_fn(values)
    func = fnx.init(func, update_fn)
    return func

def load_pytorch(file: str):
    """Loads weights from a file saved using PyTorch. Transposes weights to align with Haiku.

    Args:
        file: File to load weights from.

    Returns:
        A dictionary ``{name: weight}`` with loaded weights.
    """

    import torch
    pth = torch.load(file, map_location=torch.device("cpu"))
    if "state_dict" in pth:
        pth = pth["state_dict"]
    elif "model_state" in pth:
        pth = pth["model_state"]
    elif "model" in pth:
        pth = pth["model"]

    pth = {k: np.asarray(v) for k, v in pth.items() if not k.endswith("num_batches_tracked")}

    def preprocess_weight(name, value):
        if (name.endswith(".weight") or name == "weight") and len(value.shape) >= 2 or (name.endswith(".in_proj_weight") and len(value.shape) == 2):
            value = np.transpose(value, list(range(2, len(value.shape))) + [1, 0])
        return np.asarray(value)
    pth = {name: preprocess_weight(name, value) for name, value in pth.items()}

    return pth

def load_tensorflow(file):
    """Loads weights from a file saved using Tensorflow.

    Args:
        file: File to load weights from.

    Returns:
        A dictionary ``{name: weight}`` with loaded weights.
    """

    import tensorflow as tf
    ckpt = tf.train.load_checkpoint(file)
    ckpt_names = list(ckpt.get_variable_to_shape_map().keys())
    ckpt = {n: np.asarray(ckpt.get_tensor(n)) for n in ckpt_names}

    return ckpt

def load_numpy(file):
    """Loads weights from a file saved using Numpy.

    Args:
        file: File to load weights from.

    Returns:
        A dictionary ``{name: weight}`` with loaded weights.
    """
    npz = np.load(file)
    npz = dict(npz)
    return npz

def load_jittor(file):
    """Loads weights from a file saved using Jittor. Transposes weights to align with Haiku.

    Args:
        file: File to load weights from.

    Returns:
        A dictionary ``{name: weight}`` with loaded weights.
    """
    if file.endswith(".pkl"):
        # See: https://github.com/Jittor/jittor/blob/607d13079f27dd3a701a2c6b7a0236c66bba0699/python/jittor/__init__.py#L81
        import hashlib, pickle
        with open(file, "rb") as f:
            s = f.read()
        if s.endswith(b"HCAJSLHD"):
            checksum = s[-28:-8]
            s = s[:-28]
            if hashlib.sha1(s).digest() != checksum:
                raise ValueError(f"Pickle checksum does not match! path: {file}\nThis file maybe corrupted, please consider remove it and re-download.")
        try:
            pkl = pickle.loads(s)
        except Exception as e:
            msg = str(e)
            msg += f"\nPath: \"{file}\""
            if "trunc" in msg:
                msg += "\nThis file maybe corrupted, please consider remove it and re-download."
            raise RuntimeError(msg)
    else:
        raise ValueError(f"Invalid file format {file}")

    pkl = {k: np.asarray(v) for k, v in pkl.items() if not k.endswith("num_batches_tracked")}

    def preprocess(name, value):
        if (name.endswith(".weight") or name == "weight") and len(value.shape) >= 2:
            value = np.transpose(value, list(range(2, len(value.shape))) + [1, 0])
        return value
    pkl = {name: preprocess(name, value) for name, value in pkl.items()}

    return pkl




path = os.path.expanduser("~/.fnx")
timeout = 60.0

def download(url, file=None, timeout=timeout):
    """Downloads the file from the given url into the folder ~/.fnx.

    Args:
        url: URL to download.
        file: Optional filename to save the file. If ``None``, determines the filename from the url. Defaults to ``None``.
        timeout: Timeout in seconds for the download. Defaults to 60.0.

    Returns:
        Returns the path to the downloaded file.
    """

    if url.startswith("https://huggingface.co/"):
        # Huggingface download
        import huggingface_hub
        match = re.match(re.escape("https://huggingface.co/") + "(.+?)" + re.escape("/blob/main/") + "(.+)", url)
        assert match
        repo_id = match.group(1)
        file = match.group(2)
        file = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=file,
            etag_timeout=timeout,
        )
        return file
    else:
        # Regular download
        if file is None:
            file = url.split("/")[-1]
        if not os.path.isabs(file):
            if not os.path.isdir(path):
                os.makedirs(path)
            file = os.path.join(path, file)

        if os.path.isfile(file):
            return file

        resp = requests.get(url, stream=True, timeout=timeout)
        total = int(resp.headers.get("content-length", 0))
        received = 0
        with open(file, "wb") as f, tqdm.tqdm(desc="Downloading " + os.path.basename(file), total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
                received += size
        if received < total:
            raise requests.exceptions.RequestException("Content too short", response=resp)
        return file

def download_googledrive(file, googledrive_id, timeout=timeout):
    """Downloads the file from Google Drive into the folder ~/.fnx.

    Args:
        file: Filename for the downloaded file
        googledrive_id: Identifier of the Google Drive file to download. Can be extracted from the url: https://drive.google.com/uc?id={googledrive_id}
        timeout: Timeout in seconds for the download. Defaults to 60.0.

    Returns:
        Returns the path to the downloaded file.
    """

    if not os.path.isabs(file):
        if not os.path.isdir(path):
            os.makedirs(path)
        file = os.path.join(path, file)

    if os.path.isfile(file):
        return file

    import gdown
    if gdown.download(output=file, id=googledrive_id, quiet=False) is None:
        print(f"Failed to download from google drive. Please manually download the file https://drive.google.com/uc?id={googledrive_id} to {file}")
        sys.exit(-1)

    return file

def download_onedrive(file, url, timeout=timeout):
    """Download a file from OneDrive into the folder ~/.fnx.

    Currently does not work. Prints a request for the user to manually download the file, if it is not yet downloaded.

    Args:
        file: Filename for the downloaded file
        url: URL to the OneDrive file.
        timeout: Timeout in seconds for the download. Defaults to 60.0.

    Returns:
        Returns the path to the downloaded file.
    """

    if not os.path.isabs(file):
        if not os.path.isdir(path):
            os.makedirs(path)
        file = os.path.join(path, file)

    if os.path.isfile(file):
        return file

    print(f"Please download the file from the following link and move it to {file}:\n{url}\n")
    sys.exit(-1)
