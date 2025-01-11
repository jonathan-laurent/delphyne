//////
/// Code actions on demonstrations
//////

import * as vscode from "vscode";
import { DemosManager } from "./demos_manager";
import { DelphyneServer } from "./server";
import { StrategyDemoElement } from "./elements";
import { DemoFeedback, StrategyDemoFeedback } from "./stubs/feedback";
import { ExecutionContext, getExecutionContext } from "./execution_contexts";
import { StrategyDemo } from "./stubs/demos";
import { PointedTree, TreeInfo, TreeView } from "./tree_view";
import { ROOT_ID } from "./common";

export class DemosActionsProvider implements vscode.CodeActionProvider {
  constructor(
    private demosManager: DemosManager,
    private treeView: TreeView,
    private server: DelphyneServer,
  ) {}

  public provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range,
  ): vscode.CodeAction[] | undefined {
    // TODO: actions for standalone queries
    const element = this.demosManager.getElementAt(document.uri, range.start);
    if (element && element.kind === "strategy_demo") {
      const exe = getExecutionContext();
      const evaluateDemo = this.evaluateDemoAction(element, exe);
      const viewTreeRoot = this.viewTreeRoot(element);
      const viewTestDestination = this.viewTestDestination(element);
      const seePrompt = this.answerQuery(element, true);
      const answerQuery = this.answerQuery(element, false);
      const all: (vscode.CodeAction | null)[] = [
        seePrompt,
        answerQuery,
        viewTestDestination,
        viewTreeRoot,
        evaluateDemo,
      ];
      return all.filter((a): a is vscode.CodeAction => a !== null);
    }
    return [];
  }

  public register(context: vscode.ExtensionContext) {
    context.subscriptions.push(
      vscode.languages.registerCodeActionsProvider("yaml", this),
      vscode.commands.registerCommand("delphyne.evaluateDemo", evaluateDemo),
      vscode.commands.registerCommand(
        "delphyne.openPointedTree",
        openPointedTree,
      ),
      vscode.commands.registerCommand("delphyne.evaluateAllDemos", async () => {
        await evaluateAllDemos(this.server, this.demosManager);
      }),
    );
  }

  private evaluateDemoAction(
    element: StrategyDemoElement,
    exeContext: ExecutionContext,
  ): vscode.CodeAction {
    const action = new vscode.CodeAction(
      "Evaluate Demonstration",
      vscode.CodeActionKind.Empty,
    );
    action.command = {
      command: "delphyne.evaluateDemo",
      title: "Evaluate Demonstration",
      arguments: [element, exeContext, this.server, this.demosManager],
    };
    return action;
  }

  private viewTreeRoot(element: StrategyDemoElement): vscode.CodeAction | null {
    const feedback = this.demosManager.getFeedback(
      element.uri,
      element.demo,
    ) as StrategyDemoFeedback;
    if (feedback && ROOT_ID in feedback.trace.nodes) {
      const action = new vscode.CodeAction(
        "View Tree Root",
        vscode.CodeActionKind.Empty,
      );
      const tree = new TreeInfo(feedback.trace, element);
      const pointedTree = new PointedTree(tree, ROOT_ID);
      action.command = {
        command: "delphyne.openPointedTree",
        title: "Open Pointed Tree",
        arguments: [pointedTree, this.treeView],
      };
      return action;
    }
    return null;
  }

  private viewTestDestination(
    element: StrategyDemoElement,
  ): vscode.CodeAction | null {
    const specific = element.specific;
    if (!specific || specific.kind !== "test") {
      return null;
    }
    const feedback = this.demosManager.getFeedback(
      element.uri,
      element.demo,
    ) as StrategyDemoFeedback | null;
    if (!feedback) {
      return null;
    }
    const node_id = feedback.test_feedback[specific.test_index].node_id;
    if (node_id !== null && node_id in feedback.trace.nodes) {
      const action = new vscode.CodeAction(
        "View Test Destination",
        vscode.CodeActionKind.Empty,
      );
      const tree = new TreeInfo(feedback.trace, element);
      const pointedTree = new PointedTree(tree, node_id);
      action.command = {
        command: "delphyne.openPointedTree",
        title: "Open Pointed Tree",
        arguments: [pointedTree, this.treeView],
      };
      return action;
    }
    return null;
  }

  private answerQuery(
    element: StrategyDemoElement,
    promptOnly: boolean,
  ): vscode.CodeAction | null {
    const specific = element.specific;
    if (!specific || specific.kind !== "query") {
      return null;
    }
    const demo = JSON.parse(element.demo) as StrategyDemo;
    const query = demo.queries[specific.query_index];
    const args = {
      query: query.query,
      completions: 1,
      prompt_only: promptOnly,
      params: {},
      options: {},
      args: query.args,
    };
    const execute = promptOnly;
    const cmd = { command: "answer_query", args };
    const title = promptOnly ? "See Prompt" : "Answer Query";
    const action = new vscode.CodeAction(title, vscode.CodeActionKind.Empty);
    action.command = {
      command: "delphyne.createCommandBuffer",
      title,
      arguments: [cmd, execute],
    };
    return action;
  }
}

//////
/// Commands
//////

async function evaluateDemo(
  element: StrategyDemoElement,
  executionContext: ExecutionContext,
  server: DelphyneServer,
  demosManager: DemosManager,
) {
  const demo = JSON.parse(element.demo) as StrategyDemo;
  const task = server.launchTask(
    {
      name: "evaluate",
      arg: demo.strategy,
      origin: element,
    },
    {},
    "demo-feedback",
    { demo, context: executionContext },
  );
  const outcome = await task.outcome;
  const feedback = outcome.result as DemoFeedback;
  // The world may have changed a lot when we arrive here!
  demosManager.receiveFeedback(element.uri, element.demo, feedback);
}

async function evaluateAllDemos(
  server: DelphyneServer,
  demosManager: DemosManager,
) {
  const exe = getExecutionContext();
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
  const elements = demosManager.getAllDemoElements(editor.document.uri);
  if (!elements) {
    return;
  }
  for (const element of elements) {
    if (element.kind === "strategy_demo") {
      await evaluateDemo(element, exe, server, demosManager);
    }
  }
}

function openPointedTree(pointedTree: PointedTree, treeView: TreeView) {
  treeView.setPointedTree(pointedTree);
}
