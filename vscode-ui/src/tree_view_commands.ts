//////
/// Tree View Commands
//////

import * as vscode from "vscode";
import {
  NodeViewItem,
  QueryKey,
  TreeView,
  demoQueryKey,
  PropertyItem,
  queryKey,
  AnswerItem,
  copiableValue,
} from "./tree_view";
import { DemosManager } from "./demos_manager";
import {
  Query,
  StrategyDemoFeedback,
  TraceAnswerId,
  TraceNodeId,
} from "./stubs/feedback";
import { log } from "./logging";
import { getEditorForUri, insertYamlListElement } from "./edit_utils";
import { prettyYaml } from "./yaml_utils";
import { StrategyDemo } from "./stubs/demos";
import { Element } from "./elements";
import { gotoCommandResultAnswer } from "./commands";
import { addQuery } from "./edit";

export function registerTreeViewCommands(
  context: vscode.ExtensionContext,
  treeView: TreeView,
  demosManager: DemosManager,
) {
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "delphyne.addQueryFromTreeView",
      (arg: NodeViewItem) => {
        const prop_item = arg as PropertyItem;
        const query = prop_item.prop as Query;
        addQueryFromTreeView(query, demosManager, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.gotoQuery",
      (arg: NodeViewItem) => {
        const prop_item = arg as PropertyItem;
        const query = prop_item.prop as Query;
        gotoQuery(query, demosManager, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.gotoAnswer",
      (arg: NodeViewItem) => {
        const answer_item = arg as AnswerItem;
        gotoAnswer(answer_item.answer_id, demosManager, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.viewNode",
      (arg: { node_id: TraceNodeId }) => {
        viewNode(arg.node_id, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.copyArgValue",
      async (arg: any) => {
        const toCopy = copiableValue(arg);
        await vscode.env.clipboard.writeText(toCopy);
      },
    ),
  );
}

function getDemoInfoForTreeView(
  demosManager: DemosManager,
  treeView: TreeView,
): [StrategyDemo, StrategyDemoFeedback, vscode.TextEditor] | undefined {
  const origin = treeView.getPointedTree()?.tree.origin;
  if (!origin || origin.kind !== "strategy_demo") {
    log.error(
      "delphyne.addQueryFromTreeView: the tree view is not attached to a demo.",
    );
    return;
  }
  const demonstration = demosManager.getDemonstration(
    origin.uri,
    origin.demo,
  ) as StrategyDemo;
  if (!demonstration) {
    log.error("delphyne.addQueryFromTreeView: failed to obtain demonstration.");
    return;
  }
  const feedback = demosManager.getFeedback(
    origin.uri,
    origin.demo,
  ) as StrategyDemoFeedback;
  if (!feedback) {
    log.error("delphyne.addQueryFromTreeView: failed to obtain feedback.");
    return;
  }
  const editor = getEditorForUri(origin.uri);
  if (!editor) {
    log.warn("No editor associated with", origin.uri);
    return;
  }
  return [demonstration, feedback, editor];
}

function addQueryFromTreeView(
  query: Query,
  demosManager: DemosManager,
  treeView: TreeView,
) {
  const demoInfo = getDemoInfoForTreeView(demosManager, treeView);
  if (!demoInfo) {
    return;
  }
  const [demonstration, _, editor] = demoInfo;
  addQuery(query, demonstration, editor);
}

function queryIndex(key: QueryKey, demo: StrategyDemo): number {
  return demo.queries.findIndex((q) => demoQueryKey(q) === key);
}

function gotoQuery(
  query: Query,
  demosManager: DemosManager,
  treeView: TreeView,
) {
  const demoInfo = getDemoInfoForTreeView(demosManager, treeView);
  if (!demoInfo) {
    return;
  }
  const [demonstration, _, editor] = demoInfo;
  const idx = queryIndex(queryKey(query), demonstration);
  if (idx < 0) {
    log.error("Failed to find the query index to jump to.");
    return;
  }
  const range = demonstration.__loc_items__queries[idx];
  editor.selection = new vscode.Selection(range.start, range.end);
  editor.revealRange(range);
}

function gotoAnswer(
  answer: TraceAnswerId,
  demosManager: DemosManager,
  treeView: TreeView,
) {
  if (treeView.getPointedTree()?.tree.origin?.kind === "command") {
    const origin = treeView.getPointedTree()?.tree.origin as Element;
    gotoCommandResultAnswer(origin.uri, answer);
    return;
  }
  const demoInfo = getDemoInfoForTreeView(demosManager, treeView);
  if (!demoInfo) {
    return;
  }
  const [demonstration, feedback, editor] = demoInfo;
  const [queryIdx, answerIdx] = feedback.answer_refs[answer];
  const range = demonstration.queries[queryIdx].__loc_items__answers[answerIdx];
  editor.selection = new vscode.Selection(range.start, range.end);
  editor.revealRange(range);
}

function viewNode(node_id: TraceNodeId, treeView: TreeView) {
  treeView.setSelectedNode(node_id);
}
