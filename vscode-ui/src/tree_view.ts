//////
/// The Delphyne Tree View
//////

import * as vscode from "vscode";
import {
  Node,
  NodeProperty,
  Action,
  Trace,
  Query,
  ValueRepr,
  TraceAnswerId,
  TraceNodeId,
  NestedTree,
} from "./stubs/feedback";
import { Element } from "./elements";
import { QueryDemo, StrategyDemo } from "./stubs/demos";
import {
  prettyYaml,
  prettyYamlOneLiner,
  serializeWithoutLocInfo,
} from "./yaml_utils";
import { ROOT_ID } from "./common";

const USE_PROPERTY_ICONS = true;
const COLLAPSE_BY_DEFAULT = true;
const DEFAULT_LABEL = "' '"; // alt: __default__
const ANON_LABEL = ""; // alt: __anon__
const LEAVES_LABEL = "results";
const QUOTE_HINTS = true;
const SUCCESS_ARG_LABEL = "success";

const SUCCESS_ICON: "pass" | "check" = "pass";
const FAILURE_ICON: "close" | "error" = "error";
const MAYBE_SUCCESS_ICON = "question";
const QUERY_ICON = "comment";
const NESTED_TREE_ICON = "type-hierarchy-super";
const NODE_ICON = "circle-outline";
const SELECTED_NODE_ICON = "circle-filled";

function renderActionHints(hints: string[]): string {
  if (hints.length === 0) {
    return DEFAULT_LABEL;
  }
  const s = hints.join(" ");
  return QUOTE_HINTS ? `'${s}'` : s;
}

function renderAnswerHint(
  hint: [] | [string] | null,
  answer_id: TraceAnswerId,
) {
  return hint === null
    ? `@${answer_id}`
    : hint.length === 0
      ? DEFAULT_LABEL
      : `'${hint[0]}'`;
}

function renderActionLabel(main: string, numDescendants: number): string {
  return `${main} / ${numDescendants}`;
}

// function renderActionLabel(main: string, numDescendants: number): string {
//   return `(${numDescendants}) ${main}`;
// }

const MAYBE_EXPAND = COLLAPSE_BY_DEFAULT
  ? vscode.TreeItemCollapsibleState.Collapsed
  : vscode.TreeItemCollapsibleState.Expanded;

const COPIABLE_CONTEXT_VALUE = "copiable";
const NODE_CONTEXT_VALUE = "node";

// All the information necessary to visualize a tree.
export class TreeInfo {
  constructor(
    public readonly trace: Trace,
    public readonly origin: Element,
  ) {
    this.cached = computeCachedInfo(trace, origin);
  }
  public readonly cached: TreeCachedInfo;
}

// Information used to visualize a tree, which can be computed once from the server
// feedback and cached.
interface TreeCachedInfo {
  fromDemo: boolean;
  existingQueries: Set<QueryKey>;
}

// The input of a tree view, which is determined by a tree along with a node within it.
export class PointedTree {
  constructor(
    public readonly tree: TreeInfo,
    public readonly selectedNode: number,
  ) {}

  getNode(): Node {
    return this.tree.trace.nodes[this.selectedNode];
  }
}

//////
/// Computing the cache
//////

export type QueryKey = string;

export function queryKey(query: Query): QueryKey {
  return serializeWithoutLocInfo({ name: query.name, args: query.args });
}

export function queryDemoKey(query: QueryDemo): QueryKey {
  return serializeWithoutLocInfo({ name: query.query, args: query.args });
}

function computeCachedInfo(trace: Trace, origin: Element): TreeCachedInfo {
  let existingQueries: QueriesMap = new Set();
  if (origin.kind === "strategy_demo") {
    const demo = JSON.parse(origin.demo) as StrategyDemo;
    for (const query of demo.queries) {
      existingQueries.add(queryDemoKey(query));
    }
  }
  return { existingQueries, fromDemo: origin.kind === "strategy_demo" };
}

type QueriesMap = Set<QueryKey>;

function queryContextValue(query: Query, pointedTree: PointedTree) {
  const key = queryKey(query);
  const cached = pointedTree.tree.cached;
  let exists: boolean | null = null;
  if (cached.existingQueries.has(key)) {
    exists = true;
  } else if (cached.fromDemo) {
    exists = false;
  }
  return `query:exists-${exists}`;
}

//////
/// Visualizing Python Values
//////

interface OpaqueObject {
  kind: "opaque";
  short_descr: string;
  long_descr: string;
}

type StructuredObject = {
  kind: "object";
  value: unknown;
  short_descr: string | null;
};

export type ObjectItem = OpaqueObject | StructuredObject;

export type ArgItem = {
  kind: "arg";
  arg: string;
  value: ObjectItem;
};

export function clipboardValueOfObjectItem(item: ObjectItem): string {
  if (item.kind === "opaque") {
    return item.long_descr;
  } else {
    const value = item.value;
    if (typeof value === "string") {
      return value;
    }
    return prettyYaml(item.value);
  }
}

function objectItemOfValueRepr(repr: ValueRepr): ObjectItem {
  if (repr.json_provided) {
    return { kind: "object", value: repr.json, short_descr: repr.short };
  } else {
    return {
      kind: "opaque",
      short_descr: repr.short,
      long_descr: repr.long ?? "",
    };
  }
}

function argItemsOfDict(args: { [key: string]: ValueRepr }): ArgItem[] {
  return Object.entries(args).map(([k, v]) => {
    return { kind: "arg", arg: k, value: objectItemOfValueRepr(v) };
  });
}

function childrenOfObjectItem(item: ObjectItem): ArgItem[] {
  if (item.kind === "opaque") {
    return [];
  } else {
    let value = item.value;
    if (typeof value !== "object" || value === null) {
      return [];
    }
    return Object.entries(value).map(([k, v]) => {
      return {
        kind: "arg",
        arg: value instanceof Array ? "-" : k,
        value: { kind: "object", value: v, short_descr: null },
      };
    });
  }
}

function childrenOfArgItem(argItem: ArgItem): ArgItem[] {
  return childrenOfObjectItem(argItem.value);
}

function treeItemOfObjectItem(
  label: string,
  obj: ObjectItem,
  showIcon: boolean = false,
): vscode.TreeItem {
  let item = new vscode.TreeItem(label);
  if (obj.kind === "opaque") {
    item.description = obj.short_descr;
    item.iconPath = new vscode.ThemeIcon("symbol-misc");
  } else {
    let value = obj.value;
    item.description = prettyYamlOneLiner(value);
    // item.description = obj.short_descr ?? prettyYamlOneLiner(value);
    item.tooltip = prettyYaml(value).trim();
    switch (typeof value) {
      case "string":
        item.iconPath = new vscode.ThemeIcon("symbol-string");
        item.description = value;
        break;
      case "number":
        item.iconPath = new vscode.ThemeIcon("symbol-number");
        break;
      case "object":
        if (value === null) {
          item.iconPath = new vscode.ThemeIcon("symbol-null");
        } else if (value instanceof Array) {
          item.iconPath = new vscode.ThemeIcon("symbol-array");
          item.collapsibleState = vscode.TreeItemCollapsibleState.Collapsed;
        } else {
          item.iconPath = new vscode.ThemeIcon("symbol-object");
          item.collapsibleState = vscode.TreeItemCollapsibleState.Collapsed;
        }
        break;
    }
  }
  if (!showIcon) {
    item.iconPath = undefined;
  }
  item.contextValue = COPIABLE_CONTEXT_VALUE;
  return item;
}

function treeItemOfArgItem(argItem: ArgItem): vscode.TreeItem {
  return treeItemOfObjectItem(argItem.arg, argItem.value);
}

type ArgsItem = {
  kind: "args";
  items: ArgItem[];
};

function treeItemOfArgsItem(argsItem: ArgsItem): vscode.TreeItem {
  const item = new vscode.TreeItem("arguments", MAYBE_EXPAND);
  item.contextValue = COPIABLE_CONTEXT_VALUE;
  return item;
}

function childrenOfArgsItem(argsItem: ArgsItem): ArgItem[] {
  return argsItem.items;
}

export function copiableValue(item: ArgItem | ArgsItem): string {
  if (item.kind === "arg") {
    return clipboardValueOfObjectItem(item.value);
  } else {
    // Ensure that no argument is opaque. Then, make an object that collects then and  all
    // clipboardValueOfObjectItem. Otherwise, return an error.
    const args = item.items;
    const opaqueArg = args.find((arg) => arg.value.kind === "opaque");
    if (opaqueArg) {
      return `Error: the argument ${opaqueArg.arg} is opaque.`;
    }
    const obj: { [key: string]: unknown } = {};
    for (const arg of args) {
      obj[arg.arg] = (arg.value as StructuredObject).value;
    }
    return clipboardValueOfObjectItem({
      kind: "object",
      value: obj,
      short_descr: null,
    });
  }
}

//////
/// Misc node types
//////

export type AnswerItem = {
  kind: "answer";
  hint: [string] | [] | null;
  answer_id: TraceAnswerId;
  value: ObjectItem;
};

function treeItemOfAnswerItem(answerItem: AnswerItem): vscode.TreeItem {
  const label = renderAnswerHint(answerItem.hint, answerItem.answer_id);
  const item = treeItemOfObjectItem(label, answerItem.value);
  item.contextValue = "answer";
  return item;
}

function childrenOfAnswerItem(answerItem: AnswerItem): ArgItem[] {
  return childrenOfObjectItem(answerItem.value);
}

type AnswersItem = {
  kind: "answers";
  items: AnswerItem[];
};

function treeItemOfAnswersItem(answersItem: AnswersItem): vscode.TreeItem {
  return new vscode.TreeItem("answers", MAYBE_EXPAND);
}

function childrenOfAnswersItem(answersItem: AnswersItem): AnswerItem[] {
  return answersItem.items;
}

type SuccessItem = {
  kind: "success";
  node_id: TraceNodeId;
  label: string;
  value: ObjectItem;
};

type FailureItem = {
  kind: "failure";
  node_id: TraceNodeId;
  label: string;
  message: string;
};

type LeavesItem = {
  kind: "leaves";
  leaves: (SuccessItem | FailureItem)[];
};

function treeItemOfSuccessItem(success: SuccessItem): vscode.TreeItem {
  const item = treeItemOfObjectItem(success.label, success.value);
  item.contextValue = NODE_CONTEXT_VALUE;
  item.iconPath = new vscode.ThemeIcon(SUCCESS_ICON);
  return item;
}

function childrenOfSuccessItem(success: SuccessItem): ArgItem[] {
  return childrenOfObjectItem(success.value);
}

function treeItemOfFailureItem(failure: FailureItem): vscode.TreeItem {
  const item = new vscode.TreeItem(failure.label);
  item.description = failure.message;
  item.contextValue = NODE_CONTEXT_VALUE;
  item.tooltip = failure.message;
  item.iconPath = new vscode.ThemeIcon(FAILURE_ICON); // alternative: `close`
  return item;
}

function childrenOfFailureItem(failure: FailureItem): ArgItem[] {
  return [];
}

function childrenOfLeavesItem(
  leavesItem: LeavesItem,
): (SuccessItem | FailureItem)[] {
  return leavesItem.leaves;
}

function treeItemOfLeavesItem(leavesItem: LeavesItem): vscode.TreeItem {
  return new vscode.TreeItem(LEAVES_LABEL, MAYBE_EXPAND);
}

//////
/// Leaf Finder
//////

function findLeaves(
  trace: Trace,
  node_id: TraceNodeId,
  hints_acc: string[] | null,
): (SuccessItem | FailureItem)[] {
  const node = trace.nodes[node_id];
  if (node.leaf_node) {
    const label =
      hints_acc === null ? `%${node_id}` : renderActionHints(hints_acc);
    if (node.success_value !== null) {
      return [
        {
          kind: "success",
          node_id,
          label: label,
          value: objectItemOfValueRepr(node.success_value),
        },
      ];
    } else {
      return [
        {
          kind: "failure",
          node_id,
          label: label,
          message: node.summary_message ?? "Failure",
        },
      ];
    }
  }
  let ret: (SuccessItem | FailureItem)[] = [];
  for (const action of node.actions) {
    const dest = action.destination;
    const new_hints_acc =
      action.hints !== null && hints_acc !== null
        ? hints_acc.concat(action.hints)
        : null;
    ret = ret.concat(findLeaves(trace, dest, new_hints_acc));
  }
  return ret;
}

//////
/// Counting descendants
//////

function collectDescendants(
  trace: Trace,
  nodeId: TraceNodeId,
  acc: Set<TraceNodeId>,
): void {
  acc.add(nodeId);
  const node = trace.nodes[nodeId];
  for (const action of node.actions) {
    collectDescendants(trace, action.destination, acc);
  }
  for (const [_, nestedTree] of node.properties) {
    if (nestedTree.kind === "nested" && nestedTree.node_id !== null) {
      collectDescendants(trace, nestedTree.node_id, acc);
    }
  }
}

function countDescendants(trace: Trace, nodeId: TraceNodeId): number {
  let descendants: Set<TraceNodeId> = new Set();
  collectDescendants(trace, nodeId, descendants);
  return descendants.size;
}

//////
/// Tree View
//////

export class TreeView {
  // The pointed tree that is being visualized
  private pointedTree: PointedTree | null = null;
  // Navigation history
  private navigationHistory: TraceNodeId[] = [];
  // View providers
  private pathViewProvider: PathView = new PathView(this);
  private nodeViewProvider: NodeView = new NodeView(this);
  private actionsViewProvider: ActionsView = new ActionsView(this);
  // Views
  private pathView = vscode.window.createTreeView<PathViewItem>(
    "delphyne.tree.path",
    { treeDataProvider: this.pathViewProvider },
  );
  private nodeView = vscode.window.createTreeView<NodeViewItem>(
    "delphyne.tree.node",
    { treeDataProvider: this.nodeViewProvider },
  );
  private actionsView = vscode.window.createTreeView<ActionViewItem>(
    "delphyne.tree.actions",
    { treeDataProvider: this.actionsViewProvider },
  );

  constructor(context: vscode.ExtensionContext) {
    context.subscriptions.push(this.pathView, this.nodeView, this.actionsView);
    context.subscriptions.push(
      vscode.commands.registerCommand("delphyne.closeTreeView", () => {
        this.setPointedTree(null);
      }),
      vscode.commands.registerCommand(
        "delphyne.undoTreeNavigationAction",
        () => {
          this.undoNavigationAction();
        },
      ),
      vscode.commands.registerCommand("delphyne.jumpToNodeWithId", async () => {
        const value = await vscode.window.showInputBox({
          title: "Jump to node",
          prompt: "Enter a node identifier",
        });
        if (value === undefined) {
          return;
        }
        const node_id = Number(value);
        this.setSelectedNode(node_id, true);
      }),
    );
  }

  getPointedTree(): PointedTree | null {
    return this.pointedTree;
  }

  setPointedTree(pointedTree: PointedTree | null) {
    this.pointedTree = pointedTree;
    if (pointedTree) {
      this.navigationHistory = [pointedTree.selectedNode];
    } else {
      this.navigationHistory = [];
    }
    this.updateViews();
  }

  setSelectedNode(node_id: TraceNodeId, push_to_history: boolean = true) {
    if (this.pointedTree === null) {
      return;
    }
    if (push_to_history) {
      this.navigationHistory.push(node_id);
    }
    this.pointedTree = new PointedTree(this.pointedTree.tree, node_id);
    this.updateViews();
  }

  undoNavigationAction() {
    if (this.navigationHistory.length > 1) {
      this.navigationHistory.pop();
      const topElt = this.navigationHistory[this.navigationHistory.length - 1];
      this.setSelectedNode(topElt, false);
    }
  }

  private updateViews() {
    this.nodeView.title = "Node";
    this.nodeView.message = undefined;
    if (this.pointedTree) {
      const node = this.pointedTree.getNode();
      const nodeId = this.pointedTree.selectedNode;
      this.nodeView.description = `${node.kind} (${nodeId})`;
      if (node.summary_message !== null) {
        this.nodeView.message = node.summary_message;
      }
    } else {
      this.nodeView.description = undefined;
    }
    // Update data providers
    this.pathViewProvider.refresh();
    this.nodeViewProvider.refresh();
    this.actionsViewProvider.refresh();
    vscode.commands.executeCommand(
      "setContext",
      "delphyne.isDisplayingTree",
      this.pointedTree !== null,
    );
  }
}

//////
/// Node View
//////

export interface PropertyItem {
  kind: "property";
  label: string;
  prop: NodeProperty;
  node_id: TraceNodeId | null;
}

export type NodeViewItem =
  | PropertyItem
  | ArgsItem
  | ArgItem
  | AnswersItem
  | AnswerItem
  | LeavesItem
  | SuccessItem
  | FailureItem;

function spaceLabelFromNameAndTags(name: string, tags: string[]): string {
  if (tags.length == 1 && tags[0] === name) {
    return name;
  }
  return `${name} (${tags.join("&")})`;
}

class NodeView implements vscode.TreeDataProvider<NodeViewItem> {
  constructor(private parent: TreeView) {}

  private _onDidChangeTreeData = new vscode.EventEmitter<
    NodeViewItem | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: NodeViewItem): vscode.TreeItem {
    switch (element.kind) {
      case "property": {
        const item = new vscode.TreeItem(element.label);
        item.collapsibleState = MAYBE_EXPAND;
        switch (element.prop.kind) {
          case "data":
            item.description = element.prop.content;
            item.tooltip = element.prop.content;
            if (USE_PROPERTY_ICONS) {
              item.iconPath = new vscode.ThemeIcon("symbol-misc");
            }
            break;
          case "nested":
            if (USE_PROPERTY_ICONS) {
              item.iconPath = new vscode.ThemeIcon(NESTED_TREE_ICON);
            }
            item.label = spaceLabelFromNameAndTags(
              element.prop.strategy,
              element.prop.tags,
            );
            item.description = element.label;
            if (element.node_id !== null) {
              item.contextValue = NODE_CONTEXT_VALUE;
            }
            break;
          case "query":
            if (USE_PROPERTY_ICONS) {
              item.iconPath = new vscode.ThemeIcon(QUERY_ICON);
            }
            item.contextValue = queryContextValue(
              element.prop,
              this.parent.getPointedTree()!,
            );
            item.label = spaceLabelFromNameAndTags(
              element.prop.name,
              element.prop.tags,
            );
            item.description = element.label;
            break;
        }
        return item;
      }
      case "arg":
        return treeItemOfArgItem(element);
      case "args":
        return treeItemOfArgsItem(element);
      case "answers":
        return treeItemOfAnswersItem(element);
      case "answer":
        return treeItemOfAnswerItem(element);
      case "failure":
        return treeItemOfFailureItem(element);
      case "success":
        return treeItemOfSuccessItem(element);
      case "leaves":
        return treeItemOfLeavesItem(element);
    }
  }

  async getChildren(element?: NodeViewItem): Promise<NodeViewItem[]> {
    const node = this.parent.getPointedTree()?.getNode();
    if (element === undefined) {
      if (!node) {
        return [];
      }
      if (node.success_value !== null) {
        return [
          {
            kind: "arg",
            arg: SUCCESS_ARG_LABEL,
            value: objectItemOfValueRepr(node.success_value),
          },
        ];
      }
      const children: PropertyItem[] = node.properties.map(([k, v]) => {
        const node_id = v.kind === "nested" ? v.node_id : null;
        return {
          kind: "property",
          label: k.with_hints ?? k.with_ids,
          prop: v,
          node_id,
        };
      });
      return children;
    } else {
      switch (element.kind) {
        case "property": {
          if (element.prop.kind === "nested") {
            const trace = this.parent.getPointedTree()?.tree.trace;
            if (!trace || !node) {
              return [];
            }
            const args: ArgsItem = {
              kind: "args",
              items: argItemsOfDict(element.prop.args),
            };
            if (element.prop.node_id === null) {
              return [args];
            }
            const leaves: LeavesItem = {
              kind: "leaves",
              leaves: findLeaves(trace, element.prop.node_id, []),
            };
            return leaves.leaves.length > 0 ? [args, leaves] : [args];
          }
          if (element.prop.kind === "query") {
            let args: ArgItem[] = Object.entries(element.prop.args).map(
              ([k, v]) => {
                return {
                  kind: "arg",
                  arg: k,
                  value: { kind: "object", value: v, short_descr: null },
                };
              },
            );
            let all_args: ArgsItem = { kind: "args", items: args };
            let answers: AnswerItem[] = element.prop.answers.map((answer) => {
              return {
                kind: "answer",
                hint: answer.hint,
                answer_id: answer.id,
                value: objectItemOfValueRepr(answer.value),
              };
            });
            let all_answers: AnswersItem = { kind: "answers", items: answers };
            return all_answers.items.length > 0
              ? [all_args, all_answers]
              : [all_args];
          }
          return [];
        }
        case "arg":
          return childrenOfArgItem(element);
        case "args":
          return childrenOfArgsItem(element);
        case "answers":
          return childrenOfAnswersItem(element);
        case "answer":
          return childrenOfAnswerItem(element);
        case "failure":
          return childrenOfFailureItem(element);
        case "success":
          return childrenOfSuccessItem(element);
        case "leaves":
          return childrenOfLeavesItem(element);
      }
    }
  }

  refresh() {
    this._onDidChangeTreeData.fire();
  }
}

//////
/// Action View
//////

type ActionItem = {
  kind: "action";
  action: Action;
  value: ObjectItem;
  status: "success" | "failure" | "maybe";
  node_id: TraceNodeId;
  num_descendants: number;
};

type ActionViewItem = ActionItem | ArgItem;

function childrenOfActionItem(actionItem: ActionItem): ArgItem[] {
  return childrenOfObjectItem(actionItem.value);
}

function treeItemOfActionItem(actionItem: ActionItem): vscode.TreeItem {
  const action = actionItem.action;
  const labelMain =
    action.hints !== null
      ? renderActionHints(action.hints)
      : action.ref.with_ids;
  const label = renderActionLabel(labelMain, actionItem.num_descendants);
  const item = treeItemOfObjectItem(label, objectItemOfValueRepr(action.value));
  item.contextValue = NODE_CONTEXT_VALUE;
  // We do not want actions to be expanded
  item.collapsibleState = vscode.TreeItemCollapsibleState.None;
  switch (actionItem.status) {
    case "success":
      item.iconPath = new vscode.ThemeIcon(SUCCESS_ICON);
      break;
    case "failure":
      item.iconPath = new vscode.ThemeIcon(FAILURE_ICON);
      break;
    default:
      item.iconPath = new vscode.ThemeIcon(MAYBE_SUCCESS_ICON);
      break;
  }
  return item;
}

class ActionsView implements vscode.TreeDataProvider<ActionViewItem> {
  constructor(private parent: TreeView) {}
  private _onDidChangeTreeData = new vscode.EventEmitter<
    ActionViewItem | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: ActionViewItem): vscode.TreeItem {
    switch (element.kind) {
      case "action":
        return treeItemOfActionItem(element);
      case "arg":
        return treeItemOfArgItem(element);
    }
  }

  async getChildren(element?: ActionViewItem): Promise<ActionViewItem[]> {
    if (!element) {
      const node = this.parent.getPointedTree()?.getNode();
      const trace = this.parent.getPointedTree()?.tree.trace;
      const selectedNode = this.parent.getPointedTree()?.selectedNode;
      if (selectedNode === null || !node || !trace) {
        return [];
      }
      return node.actions.map((action) => {
        const leaves = findLeaves(trace, action.destination, action.hints);
        const destination = trace.nodes[action.destination];
        let status: "success" | "failure" | "maybe" = "maybe";
        if (leaves.some((leaf) => leaf.kind === "success")) {
          status = "success";
        } else if (destination.leaf_node) {
          status = "failure";
        }
        return {
          kind: "action",
          action: action,
          value: objectItemOfValueRepr(action.value),
          status: status,
          node_id: action.destination,
          num_descendants: countDescendants(trace, action.destination),
        };
      });
    } else {
      switch (element.kind) {
        case "action":
          return childrenOfActionItem(element);
        case "arg":
          return childrenOfArgItem(element);
      }
    }
  }

  refresh() {
    this._onDidChangeTreeData.fire();
  }
}

//////
/// Path View
//////

type PathViewNodeItem = {
  kind: "path_node";
  node: Node;
  node_id: TraceNodeId;
  nestedTrees: PathNestedTreeItem[];
  expanded: boolean;
  selected: boolean;
};

type PathNestedTreeItem = {
  kind: "path_nested_tree";
  nestedTrees: NestedTree;
  path: PathViewNodeItem[];
  expanded: boolean;
};

type PathViewItem = PathViewNodeItem | PathNestedTreeItem;

// Vscode tries to be too samrt and remember which node items were collapsed between each
// view refresh. By default, it tries to infer this information from labels but it gets it
// wrong for the path view. Thus, we make sure to assign fresh ids to the path view items
// every time this view is refreshed.
let _globalPathViewItemId = 0;
function uniquePathViewItemId(): string {
  return `path_view_item_${_globalPathViewItemId++}`;
}

function treeItemOfPathViewNodeItem(
  nodeItem: PathViewNodeItem,
): vscode.TreeItem {
  const node = nodeItem.node;
  const item = new vscode.TreeItem(node.kind);
  if (node.label) {
    item.description = node.label;
  }
  item.contextValue = NODE_CONTEXT_VALUE;
  const icon = nodeItem.selected ? SELECTED_NODE_ICON : NODE_ICON;
  item.iconPath = new vscode.ThemeIcon(icon);
  item.id = uniquePathViewItemId();
  if (nodeItem.nestedTrees.length > 0) {
    item.collapsibleState = nodeItem.expanded
      ? vscode.TreeItemCollapsibleState.Expanded
      : vscode.TreeItemCollapsibleState.Expanded;
  }
  return item;
}

function treeItemOfPathViewNestedTreeItem(
  nestedTreeItem: PathNestedTreeItem,
): vscode.TreeItem {
  const nestedTree = nestedTreeItem.nestedTrees;
  const item = new vscode.TreeItem(nestedTree.strategy);
  item.iconPath = new vscode.ThemeIcon(NESTED_TREE_ICON);
  item.id = uniquePathViewItemId();
  if (nestedTreeItem.path.length > 0) {
    item.collapsibleState = item.collapsibleState = nestedTreeItem.expanded
      ? vscode.TreeItemCollapsibleState.Expanded
      : vscode.TreeItemCollapsibleState.Expanded;
  }
  return item;
}

function childrenOfPathViewNodeItem(
  nodeItem: PathViewNodeItem,
): PathViewItem[] {
  return nodeItem.nestedTrees;
}

function childrenOfPathViewNestedTreeItem(
  nestedTreeItem: PathNestedTreeItem,
): PathViewNodeItem[] {
  return nestedTreeItem.path;
}

function addChildToPath(
  path: PathViewNodeItem[],
  child: PathViewNodeItem,
  actionNestedTrees: PathNestedTreeItem[],
): void {
  const last = path[path.length - 1];
  if (last.nestedTrees.length === 0) {
    for (const nestedTree of actionNestedTrees) {
      last.nestedTrees.push(nestedTree);
    }
    path.push(child);
  } else if (last.nestedTrees.length === 1) {
    addChildToPath(last.nestedTrees[0].path, child, actionNestedTrees);
  } else {
    throw Error("Invalid path");
  }
}

function addNestedTreeToPath(
  path: PathViewNodeItem[],
  nestedTree: PathNestedTreeItem,
): void {
  const last = path[path.length - 1];
  if (last.nestedTrees.length === 0) {
    last.nestedTrees.push(nestedTree);
  } else if (last.nestedTrees.length === 1) {
    addNestedTreeToPath(last.nestedTrees[0].path, nestedTree);
  } else {
    throw Error("Invalid path");
  }
}

function setDefaultExpansionState(path: PathViewNodeItem[]): void {
  const last = path[path.length - 1];
  if (last.nestedTrees.length === 0) {
    last.selected = true;
  } else if (last.nestedTrees.length === 1) {
    last.expanded = true;
    last.nestedTrees[0].expanded = true;
    setDefaultExpansionState(last.nestedTrees[0].path);
  }
}

function singletonPath(nodeId: TraceNodeId, node: Node): PathViewNodeItem[] {
  const item: PathViewNodeItem = {
    kind: "path_node",
    node: node,
    node_id: nodeId,
    nestedTrees: [],
    expanded: false,
    selected: false,
  };
  return [item];
}

function computePath(
  trace: Trace,
  src_id: TraceNodeId,
  dst_id: TraceNodeId,
): PathViewNodeItem[] {
  // The resulting path starts with node `src`.
  // It ends with either:
  // - the node `dst` with no children
  // - a node `end` with a single nested tree child with a path to `dst`.
  if (src_id === dst_id) {
    return singletonPath(src_id, trace.nodes[src_id]);
  }
  const dst = trace.nodes[dst_id];
  if (dst.origin === "root") {
    throw Error("Invalid success path");
  }
  if (dst.origin[0] === "nested") {
    const [_, before_id, prop_id] = dst.origin;
    const sub_parent = trace.nodes[before_id];
    const path = computePath(trace, src_id, before_id);
    const nestedTreeItem: PathNestedTreeItem = {
      kind: "path_nested_tree",
      nestedTrees: sub_parent.properties[prop_id][1] as NestedTree,
      path: singletonPath(dst_id, dst),
      expanded: false,
    };
    addNestedTreeToPath(path, nestedTreeItem);
    return path;
  } else {
    const [_, before_id, action_id] = dst.origin;
    const before = trace.nodes[before_id];
    const action = before.actions[action_id];
    const nestedTrees: PathNestedTreeItem[] = action.related_success_nodes.map(
      (success_id) => {
        const subpath = computePath(trace, before_id, success_id);
        if (subpath.length !== 1 || subpath[0].nestedTrees.length !== 1) {
          throw Error("Invalid success path");
        }
        return subpath[0].nestedTrees[0];
      },
    );
    const item: PathViewNodeItem = {
      kind: "path_node",
      node: dst,
      node_id: dst_id,
      nestedTrees: [],
      expanded: false,
      selected: false,
    };
    const path = computePath(trace, src_id, before_id);
    addChildToPath(path, item, nestedTrees);
    return path;
  }
}

class PathView implements vscode.TreeDataProvider<PathViewItem> {
  constructor(private parent: TreeView) {}
  private _onDidChangeTreeData = new vscode.EventEmitter<
    PathViewItem | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(element: PathViewItem): vscode.TreeItem {
    switch (element.kind) {
      case "path_node":
        return treeItemOfPathViewNodeItem(element);
      case "path_nested_tree":
        return treeItemOfPathViewNestedTreeItem(element);
    }
  }

  async getChildren(element?: PathViewItem): Promise<PathViewItem[]> {
    if (!element) {
      const selectedNode = this.parent.getPointedTree()?.selectedNode;
      const trace = this.parent.getPointedTree()?.tree.trace;
      if (selectedNode === undefined || !trace) {
        return [];
      }
      const path = computePath(trace, ROOT_ID, selectedNode);
      setDefaultExpansionState(path);
      return path;
    }
    switch (element.kind) {
      case "path_node":
        return childrenOfPathViewNodeItem(element);
      case "path_nested_tree":
        return childrenOfPathViewNestedTreeItem(element);
    }
  }

  refresh() {
    this._onDidChangeTreeData.fire();
  }
}
