# Generic prefix-tree / observation-tree node structure used by the general passive
# (state-merging) learning algorithms, plus conversion to concrete AALpy automaton types.
import functools
import math
import pathlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import total_ordering
from typing import Any, TypeVar
import pydot
from copy import copy

from aalpy.automata import StochasticMealyMachine, StochasticMealyState, MooreState, MooreMachine, NDMooreState, \
    NDMooreMachine, Mdp, MdpState, MealyMachine, MealyState, Onfsm, OnfsmState
from aalpy.base import Automaton

Key = TypeVar("Key")
Val = TypeVar("Val")

OutputBehavior = str
OutputBehaviorRange = ["moore", "mealy"]

TransitionBehavior = str
TransitionBehaviorRange = ["deterministic", "nondeterministic", "stochastic"]

DataFormat = str
DataFormatRange = ["io_traces", "labeled_sequences", "traces", "tree"]

IOPair = tuple[Any, Any]
IOTrace = Sequence[IOPair]
IOExample = tuple[Sequence[Any], Any]

StateFunction = Callable[['GsmNode'], str]
TransitionFunction = Callable[['GsmNode', Any, Any], str]

unknown_output = None  # can be set to a special value if required


def intersection_iterator(a: dict[Key, Val], b: dict[Key, Val]) -> Iterator[tuple[Key, Val, Val]]:
    """
    Iterate over the key/value pairs that are present in both dictionaries.

    :param dict[Key, Val] a: First dictionary.
    :param dict[Key, Val] b: Second dictionary.
    :return Iterator[tuple[Key, Val, Val]]: Iterator of (key, value in a, value in b) for keys common to both dicts.
    """
    missing = object()
    for key, a_val in a.items():
        b_val = b.get(key, missing)
        if b_val is missing:
            continue
        yield key, a_val, b_val


def union_iterator(a: dict[Key, Val], b: dict[Key, Val], default: Val = None) -> Iterator[tuple[Key, Val, Val]]:
    """
    Iterate over the key/value pairs present in either dictionary, substituting a default for missing values.

    :param dict[Key, Val] a: First dictionary.
    :param dict[Key, Val] b: Second dictionary.
    :param Val default: Value used in place of a missing entry.
    :return Iterator[tuple[Key, Val, Val]]: Iterator of (key, value in a, value in b) for keys in either dict.
    """
    for key, a_val in a.items():
        b_val = b.get(key, default)
        yield key, a_val, b_val
    for key, b_val in b.items():
        if key in a:
            continue
        a_val = a.get(key, default)
        yield key, a_val, b_val


# TODO reuse in RPNI
def detect_data_format(data: Any, check_consistency: bool = False, guess: bool = False) -> DataFormat:
    """
    Guess the data format of the provided learning data.

    :param Any data: Input data: a GsmNode (tree), or a sequence of traces/examples.
    :param bool check_consistency: Whether to check all data points instead of returning as soon as a unique format is found.
    :param bool guess: Whether to allow guessing a single format when multiple formats remain ambiguous.
    :return DataFormat: The detected data format string (see DataFormatRange).
    """
    # The different data formats are
    # - "tree": a tree-shaped automaton provided as a GsmNode
    # - "io_traces": either
    #   - Moore traces [[o, (i,o), (i,o), ...], ...]
    #   - Mealy traces [[(i,o), (i,o), ...], ...]
    # - "labeled_sequences": [([i, i, ...], o), ...]
    # - "traces": [[o, o, ...], ...]

    if isinstance(data, GsmNode):
        return "tree"

    accepted_types = (tuple, list)

    # mapping data formats to compatibility criteria
    check_dict = dict(
        io_traces=lambda obj: len(obj) <= 1 or all(isinstance(o, accepted_types) and len(o) == 2 for o in obj[1:]),
        labeled_sequences=lambda obj: len(obj) == 2 and isinstance(obj[0], accepted_types),
    )
    accept_dict = {k: True for k in check_dict}

    if not isinstance(data, accepted_types):
        raise ValueError("wrong input format. expected tuple or list.")
    if len(data) == 0:
        return "io_traces"

    accepted_formats = list(accept_dict.keys())
    for data_point in data:
        if not isinstance(data_point, accepted_types):
            raise ValueError("wrong input format. expected tuple or list.")
        for k, check in check_dict.items():
            accept_dict[k] &= check(data_point)
        accepted_formats = [k for k, v in accept_dict.items() if v]
        if len(accepted_formats) == 1 and not check_consistency:
            return accepted_formats[0]
        if len(accepted_formats) == 0:
            return "traces" # default to traces
            #raise ValueError("invalid or inconsistent data. no options left")
    if len(accepted_formats) != 1 and not guess:
        raise ValueError("ambiguous data format. data format needs to be specified explicitly.")
    return accepted_formats[0]


# TODO maybe split this for maintainability (and perfomance?)
class TransitionInfo:
    """Stores the current and original (PTA) target node and count for a single transition."""

    __slots__ = ["target", "count", "original_target", "original_count"]

    def __init__(self, target: 'GsmNode', count: int, original_target: 'GsmNode | None',
                 original_count: int | None) -> None:
        """
        Create a transition info record.

        :param GsmNode target: Current target node of the transition.
        :param int count: Current transition count.
        :param GsmNode | None original_target: Target node in the original PTA, if any.
        :param int | None original_count: Transition count in the original PTA, if any.
        """
        self.target: 'GsmNode' = target
        self.count: int = count
        self.original_target: 'GsmNode' = original_target
        self.original_count: int = original_count


# TODO add custom pickling code that flattens the Node structure in order to circumvent running into recursion issues for large models
@total_ordering
class GsmNode:
    """
    Generic class for observably deterministic automata.

    The prefix is given as (minimal) list of IO pairs leading to that state.
    We assume an initial transition to the initial state, which has to be reflected in the prefix.
    This way, the output of the initial state for Moore machines can be encoded in its prefix.

    Transition count is preferred over state count as it allows to easily count transitions for non-tree-shaped automata
    """
    __slots__ = ['transitions', 'predecessor', 'prefix_access_pair']

    def __init__(self, prefix_access_pair: IOPair, predecessor: 'GsmNode | None' = None) -> None:
        """
        Create a node with the given prefix-access pair and predecessor.

        :param IOPair prefix_access_pair: (input, output) pair leading from the predecessor to this node.
        :param GsmNode | None predecessor: Predecessor node, or None for the root node.
        """
        # TODO try single dict
        self.transitions: defaultdict[Any, dict[Any, TransitionInfo]] = defaultdict(dict)
        self.predecessor: GsmNode = predecessor
        self.prefix_access_pair = prefix_access_pair

    def __lt__(self, other: 'GsmNode', compare_length_only: bool = False) -> bool:
        """
        Compare nodes in short-lex order: first by prefix length, then lexicographically by prefix.

        :param GsmNode other: Node to compare against.
        :param bool compare_length_only: Whether to only compare based on prefix length.
        :return bool: True if self is ordered before other.
        """
        own_l, other_l = self.get_prefix_length(), other.get_prefix_length()
        if own_l != other_l:
            return own_l < other_l
        if compare_length_only:
            return False
        own_p = self.get_prefix()
        other_p = other.get_prefix()
        try:
            return own_p < other_p
        except TypeError:
            return [str(x) for x in own_p] < [str(x) for x in other_p]

    # TODO implicit prefixes as currently implemented require O(length) time for prefix calculations (e.g. to determine the minimal blue node)
    # other options would be to have more efficient explicit prefixes such as shared list representations
    def get_prefix_length(self) -> int:
        """
        Compute the length of this node's prefix (distance from the root).

        :return int: Number of transitions from the root to this node.
        """
        node = self
        length = 0
        while node.predecessor:
            node = node.predecessor
            length += 1
        return length

    def get_prefix_output(self) -> Any:
        """
        Get the output of the prefix-access pair leading to this node.

        :return Any: The output symbol of the prefix-access pair.
        """
        return self.prefix_access_pair[1]

    def get_prefix_input(self) -> Any:
        """
        Get the input of the prefix-access pair leading to this node.

        :return Any: The input symbol of the prefix-access pair.
        """
        return self.prefix_access_pair[0]

    def resolve_unknown_prefix_output(self, value: Any) -> None:
        """
        Set the prefix output to the given value if it is currently unknown.

        :param Any value: Output value to assign if the current prefix output is unknown.
        """
        if self.get_prefix_output() is unknown_output:
            self.prefix_access_pair = (self.get_prefix_input(), value)

    def get_prefix(self, include_output: bool = True) -> list[Any]:
        """
        Compute the sequence of prefix-access pairs (or just inputs) leading from the root to this node.

        :param bool include_output: Whether to include the output alongside each input in the prefix.
        :return list[Any]: List of IO pairs (or inputs only) leading to this node.
        """
        node = self
        prefix = []
        while node.predecessor:
            symbol = node.prefix_access_pair
            if not include_output:
                symbol = symbol[0]
            prefix.append(symbol)
            node = node.predecessor
        prefix.reverse()
        return prefix

    def get_root(self) -> 'GsmNode':
        """
        Find the root node of the tree this node belongs to.

        :return GsmNode: The root node.
        """
        current = self
        while current.predecessor:
            current = current.predecessor
        return current

    def get_or_create_transitions(self, in_sym: Any) -> dict[Any, TransitionInfo]:
        """
        Get the transition dictionary for the given input symbol, creating it if necessary.

        :param Any in_sym: Input symbol.
        :return dict[Any, TransitionInfo]: Mapping of output symbol to transition info for this input.
        """
        t = self.transitions.get(in_sym)
        if t is None:
            t = dict()
            self.transitions[in_sym] = t
        return t

    def transition_iterator(self) -> Iterable[tuple[Any, Any, TransitionInfo]]:
        """
        Iterate over all outgoing transitions of this node.

        :return Iterable[tuple[Any, Any, TransitionInfo]]: Iterable of (input, output, transition info) triples.
        """
        for in_sym, transitions in self.transitions.items():
            for out_sym, node in transitions.items():
                yield in_sym, out_sym, node

    def shallow_copy(self) -> 'GsmNode':
        """
        Create a shallow copy of this node, duplicating its transition dict but keeping the same targets.

        :return GsmNode: The copied node.
        """
        node = GsmNode(self.prefix_access_pair, self.predecessor)
        for in_sym, t in self.transitions.items():
            d = dict() # appears to be faster than dict comprehension
            for out_sym, ti in t.items():
                d[out_sym] = TransitionInfo(ti.target, ti.count, ti.original_target, ti.original_count)
            node.transitions[in_sym] = d
        return node

    def get_by_prefix(self, seq: IOTrace) -> 'GsmNode | None':
        """
        Follow the given sequence of IO pairs from this node and return the node it leads to.

        :param IOTrace seq: Sequence of (input, output) pairs to follow.
        :return GsmNode | None: The reached node, or None if the sequence is not defined.
        """
        node: GsmNode = self
        for in_sym, out_sym in seq:
            if in_sym is None:  # ignore initial transition of Node.get_prefix()
                continue
            trans = node.transitions.get(in_sym)
            if trans is None:
                return None
            t_info = trans.get(out_sym)
            if t_info is None:
                return None
            node = t_info.target
        return node

    def get_all_nodes(self) -> list['GsmNode']:
        """
        Collect all nodes reachable from this node (including itself).

        :return list[GsmNode]: List of all reachable nodes.
        """
        result = [self]
        backing_set = {self}
        for state in result:
            for _, _, transition in state.transition_iterator():
                child = transition.target
                if child not in backing_set:
                    backing_set.add(child)
                    result.append(child)
        return result

    def is_tree(self) -> bool:
        """
        Check whether the structure reachable from this node is a tree (no shared/repeated nodes).

        :return bool: True if the structure is a tree.
        """
        q: list['GsmNode'] = [self]
        backing_set = {self}
        while len(q) != 0:
            current = q.pop(0)
            for _, _, transition in current.transition_iterator():
                child = transition.target
                if child in backing_set:
                    return False
                q.append(child)
                backing_set.add(child)
        return True

    def to_automaton(self, output_behavior: OutputBehavior, transition_behavior: TransitionBehavior,
                     check_behavior: bool = True, set_prefix: bool = False) -> Automaton:
        """
        Convert the tree/graph reachable from this node into a concrete AALpy automaton.

        :param OutputBehavior output_behavior: Either "moore" or "mealy".
        :param TransitionBehavior transition_behavior: Either "deterministic", "nondeterministic" or "stochastic".
        :param bool check_behavior: Whether to validate that the structure actually matches the requested behaviors.
        :param bool set_prefix: Whether to record the access prefix on each created state.
        :return Automaton: The resulting automaton instance.
        """
        nodes = self.get_all_nodes()

        if check_behavior:
            if output_behavior == "moore" and not self.is_moore():
                raise ValueError("Tried to obtain Moore machine from non-Moore structure")
            if transition_behavior == "deterministic" and not self.is_deterministic():
                raise ValueError("Tried to obtain deterministic automaton from non-deterministic structure")

        type_dict = {
            ("moore", "deterministic"): (MooreMachine, MooreState),
            ("moore", "nondeterministic"): (NDMooreMachine, NDMooreState),
            ("moore", "stochastic"): (Mdp, MdpState),
            ("mealy", "deterministic"): (MealyMachine, MealyState),
            ("mealy", "nondeterministic"): (Onfsm, OnfsmState),
            ("mealy", "stochastic"): (StochasticMealyMachine, StochasticMealyState),
        }

        AutomatonClass, StateClass = type_dict[(output_behavior, transition_behavior)]

        # create states
        state_map = dict()
        for i, node in enumerate(nodes):
            state_id = f's{i}'
            if output_behavior == "mealy":
                state = StateClass(state_id)
            elif output_behavior == "moore":
                state = StateClass(state_id, node.get_prefix_output())
            state_map[node] = state
            if set_prefix:
                if transition_behavior == "deterministic":
                    state.prefix = tuple(p[0] for p in node.get_prefix())
                else:
                    state.prefix = tuple(node.get_prefix())
            else:
                state.prefix = None

        initial_state = state_map[self]

        # add transitions
        for node in nodes:
            state = state_map[node]
            for in_sym, transitions in node.transitions.items():
                total = sum(t.count for t in transitions.values())
                for out_sym, target_node in transitions.items():
                    target_state = state_map[target_node.target]
                    count = target_node.count
                    if AutomatonClass is MooreMachine:
                        state.transitions[in_sym] = target_state
                    elif AutomatonClass is MealyMachine:
                        state.transitions[in_sym] = target_state
                        state.output_fun[in_sym] = out_sym
                    elif AutomatonClass is NDMooreMachine:
                        state.transitions[in_sym].append(target_state)
                    elif AutomatonClass is Onfsm:
                        state.transitions[in_sym].append((out_sym, target_state))
                    elif AutomatonClass is Mdp:
                        state.transitions[in_sym].append((target_state, count / total))
                    elif AutomatonClass is StochasticMealyMachine:
                        state.transitions[in_sym].append((target_state, out_sym, count / total))

        return AutomatonClass(initial_state, list(state_map.values()))

    def visualize(self, path: str | pathlib.Path, output_behavior: OutputBehavior = "mealy", format: str = "dot",
                  engine: str = "dot", *,
                  state_label: StateFunction | None = None, state_color: StateFunction | None = None,
                  trans_label: TransitionFunction | None = None, trans_color: TransitionFunction | None = None,
                  state_props: dict[str, StateFunction] | None = None,
                  trans_props: dict[str, TransitionFunction] | None = None,
                  node_naming: StateFunction | None = None) -> None:
        """
        Render the tree/graph reachable from this node to a graphviz file.

        :param str | pathlib.Path path: Output path (without extension).
        :param OutputBehavior output_behavior: Either "moore" or "mealy", controls default labeling.
        :param str format: Output format passed to graphviz (e.g. "dot", "pdf", "png").
        :param str engine: Graphviz layout engine to use.
        :param StateFunction | None state_label: Function computing a state's label.
        :param StateFunction | None state_color: Function computing a state's color.
        :param TransitionFunction | None trans_label: Function computing a transition's label.
        :param TransitionFunction | None trans_color: Function computing a transition's color.
        :param dict[str, StateFunction] | None state_props: Extra per-state graphviz properties, keyed by property name.
        :param dict[str, TransitionFunction] | None trans_props: Extra per-transition graphviz properties, keyed by property name.
        :param StateFunction | None node_naming: Function assigning a unique graphviz node name to each state.
        """

        # handle default parameters
        if output_behavior not in ["moore", "mealy", None]:
            raise ValueError(f"Invalid OutputBehavior {output_behavior}")
        if state_props is None:
            state_props = dict()
        if trans_props is None:
            trans_props = dict()
        if state_label is None:
            if output_behavior == "moore":
                def state_label(node: GsmNode) -> str:
                    return f'{node.get_prefix_output()} {node.count()}'
            else:
                def state_label(node: GsmNode) -> str:
                    return f'{sum(t.count for _, _, t in node.transition_iterator())}'
        if trans_label is None and "label" not in trans_props:
            if output_behavior == "moore":
                def trans_label(node: GsmNode, in_sym: Any, out_sym: Any) -> str:
                    return f'{in_sym} [{node.transitions[in_sym][out_sym].count}]'
            else:
                def trans_label(node: GsmNode, in_sym: Any, out_sym: Any) -> str:
                    return f'{in_sym} / {out_sym} [{node.transitions[in_sym][out_sym].count}]'
        if state_color is None:
            def state_color(x: 'GsmNode') -> str: return "black"
        if trans_color is None:
            def trans_color(x: 'GsmNode', y: Any, z: Any) -> str: return "black"
        if node_naming is None:
            node_dict = dict()

            def node_naming(node: GsmNode) -> str:
                if node not in node_dict:
                    node_dict[node] = f"s{len(node_dict)}"
                return node_dict[node]
        state_props = {"label": state_label, "color": state_color, "fontcolor": state_color, **state_props}
        trans_props = {"label": trans_label, "color": trans_color, "fontcolor": trans_color, **trans_props}

        # create new graph
        graph = pydot.Dot('automaton', graph_type='digraph')

        # graph.add_node(pydot.Node(str(self.prefix), label=state_label(self)))
        nodes = self.get_all_nodes()

        # add nodes
        for node in nodes:
            arg_dict = {key: fun(node) for key, fun in state_props.items()}
            graph.add_node(pydot.Node(node_naming(node), **arg_dict))

        # add transitions
        for node in nodes:
            for in_sym, options in node.transitions.items():
                for out_sym, c in options.items():
                    arg_dict = {key: fun(node, in_sym, out_sym) for key, fun in trans_props.items()}
                    graph.add_edge(pydot.Edge(node_naming(node), node_naming(c.target), **arg_dict))

        # add initial state
        # TODO maybe add option to parameterize this
        graph.add_node(pydot.Node('__start0', shape='none', label=''))
        graph.add_edge(pydot.Edge('__start0', node_naming(self), label=''))

        file_ext = format
        if format == 'dot':
            format = 'raw'
        if format == 'raw':
            file_ext = 'dot'
        graph.write(path=str(path) + "." + file_ext, prog=engine, format=format)

    def make_input_complete(self) -> list[tuple['GsmNode', Any, Any]]:
        """
        Add self-looping transitions for any input undefined at some node, using the node's prefix output.

        :return list[tuple[GsmNode, Any, Any]]: List of (node, input, output) triples for the added transitions.
        """
        all_nodes = self.get_all_nodes()
        inputs = {in_sym for node in all_nodes for in_sym in node.transitions}
        missing_trans = []
        for node in all_nodes:
            for in_sym in inputs:
                transitions = node.transitions[in_sym]
                if len(transitions) == 0:
                    out_sym = node.prefix_access_pair[1]
                    missing_trans.append((node, in_sym, out_sym))
                    t_info = TransitionInfo(node, 1, None, None)
                    transitions[out_sym] = t_info
        return missing_trans

    def add_trace(self, trace: IOTrace) -> None:
        """
        Add an IO trace to the tree rooted at this node, extending it with new nodes as necessary.

        :param IOTrace trace: Sequence of (input, output) pairs to add.
        """
        curr_node: GsmNode = self
        for in_sym, out_sym in trace:
            transitions = curr_node.transitions[in_sym]
            info = transitions.get(out_sym)
            if info is None:
                node = GsmNode((in_sym, out_sym), curr_node)
                transitions[out_sym] = TransitionInfo(node, 1, node, 1)
            else:
                info.count += 1
                info.original_count += 1
                node = info.target
            curr_node = node

    def add_labeled_sequence(self, example: IOExample) -> None:
        """
        Add a labeled input sequence (inputs with a single label attached at the end) to the tree.

        :param IOExample example: (inputs, output) pair, where output labels the state reached by inputs.
        """
        inputs, output = example
        curr_node: GsmNode = self
        in_sym = None

        # step through inputs and add transitions
        for in_sym in inputs:
            transitions = curr_node.transitions[in_sym]
            t_infos = list(transitions.values())
            if len(t_infos) == 0:
                node = GsmNode((in_sym, unknown_output), curr_node)
                t_info = TransitionInfo(node, 1, node, 1)
                transitions[unknown_output] = t_info
            elif len(t_infos) == 1:
                t_info = t_infos[0]
                t_info.count += 1
                t_info.original_count += 1
                node = t_info.target
            else:
                # This should never happen
                raise ValueError("Nondeterminism encountered for GSM with labeled_sequences. not supported")
            curr_node = node

        # set last output
        curr_node.resolve_unknown_prefix_output(output)
        pred = curr_node.predecessor
        if pred:
            transitions = pred.transitions[in_sym]
            if unknown_output in transitions:
                transitions[output] = transitions.pop(unknown_output)
            if output not in transitions:
                raise ValueError("nondeterminism encountered for GSM with labeled_sequences. not supported")

    @staticmethod
    def createPTA(data: Any, output_behavior: OutputBehavior, data_format: DataFormat | None = None) -> 'GsmNode':
        """
        Build a prefix tree acceptor (PTA) from the given data.

        :param Any data: Learning data, in one of the supported data formats (or already a GsmNode tree).
        :param OutputBehavior output_behavior: Either "moore" or "mealy".
        :param DataFormat | None data_format: Explicit data format, or None to auto-detect.
        :return GsmNode: The root node of the constructed (or passed-through) PTA.
        """
        if data_format is None:
            data_format = detect_data_format(data)
        if data_format not in DataFormatRange:
            raise ValueError(f"invalid data format {data_format}. should be in {DataFormatRange}")

        if data_format == "tree":
            if not data.is_tree():
                raise ValueError("provided automaton is not a tree")
            return data
        root_node = GsmNode((None, unknown_output), None)
        if data_format == "labeled_sequences":
            for example in data:
                root_node.add_labeled_sequence(example)
        if data_format == "io_traces" or data_format == "traces":
            if output_behavior == "moore":
                initial_output = data[0][0]
                root_node.prefix_access_pair = (None, initial_output)
                data = (d[1:] for d in data)
            for trace in data:
                if data_format == "traces":
                    trace = (("step", t) for t in trace)
                root_node.add_trace(trace)
        return root_node

    def is_locally_deterministic(self) -> bool:
        """
        Check whether this node has at most one outgoing transition per input symbol.

        :return bool: True if this node is locally deterministic.
        """
        return all(len(item) == 1 for item in self.transitions.values())

    def is_deterministic(self) -> bool:
        """
        Check whether all nodes reachable from this node are locally deterministic.

        :return bool: True if the whole structure is deterministic.
        """
        return all(node.is_locally_deterministic() for node in self.get_all_nodes())

    def deterministic_compatible(self, other: 'GsmNode') -> bool:
        """
        Check whether this node and another node have compatible outgoing input symbols (ignoring unknown outputs).

        :param GsmNode other: Node to compare against.
        :return bool: True if the two nodes are compatible with a deterministic merge.
        """
        for _, trans_self, trans_other in intersection_iterator(self.transitions, other.transitions):
            if unknown_output in trans_self or unknown_output in trans_other:
                continue
            if trans_self.keys() != trans_other.keys():
                return False
        return True

    def is_moore(self) -> bool:
        """
        Check whether the structure reachable from this node satisfies the Moore condition (output determined by state).

        :return bool: True if the structure is Moore-compatible.
        """
        for node in self.get_all_nodes():
            for in_sym, out_sym, transition in node.transition_iterator():
                child_output = transition.target.get_prefix_output()
                if out_sym is not unknown_output and child_output != out_sym:
                    return False
        return True

    def moore_compatible(self, other: 'GsmNode') -> bool:
        """
        Check whether this node and another node have compatible (or unknown) prefix outputs.

        :param GsmNode other: Node to compare against.
        :return bool: True if the prefix outputs are compatible.
        """
        so = self.get_prefix_output()
        oo = other.get_prefix_output()
        return so == oo or so is unknown_output or oo is unknown_output

    def local_log_likelihood_contribution(self) -> float:
        """
        Compute this node's contribution to the log-likelihood of the data given the model.

        :return float: The local log-likelihood contribution.
        """
        llc = 0
        for in_sym, trans in self.transitions.items():
            total_count = 0
            for out_sym, info in trans.items():
                total_count += info.count
                llc += info.count * math.log(info.count)
            if total_count != 0:
                llc -= total_count * math.log(total_count)
        return llc

    def count(self) -> int:
        """
        Compute the total transition count over all outgoing transitions of this node.

        :return int: Sum of transition counts.
        """
        return sum(trans.count for _, _, trans in self.transition_iterator())

    default_order = functools.cmp_to_key(lambda a, b: -1 if a < b else 1)
