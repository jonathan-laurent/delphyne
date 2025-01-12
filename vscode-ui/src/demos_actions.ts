//////
/// Code actions on demonstrations
//////

import * as vscode from "vscode";
import { DemosManager } from "./demos_manager";
import { DelphyneServer } from "./server";
import { DemoElement, queryOfDemoElement } from "./elements";
import { DemoFeedback, Query, StrategyDemoFeedback } from "./stubs/feedback";
import { ExecutionContext, getExecutionContext } from "./execution_contexts";
import { QueryDemo, StrategyDemo } from "./stubs/demos";
import { PointedTree, TreeInfo, TreeView } from "./tree_view";
import { ROOT_ID } from "./common";
import { addQueries } from "./edit";

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
    const element = this.demosManager.getElementAt(document.uri, range.start);
    if (element) {
      const exe = getExecutionContext();
      const evaluateDemo = this.evaluateDemoAction(element, exe);
      const viewTreeRoot = this.viewTreeRoot(element);
      const viewTestDestination = this.viewTestDestination(element);
      const seePrompt = this.answerQuery(element, true);
      const answerQuery = this.answerQuery(element, false);
      const addImplicitAnswers = this.addImplicitAnswers(element);
      const all: (vscode.CodeAction | null)[] = [
        seePrompt,
        answerQuery,
        viewTestDestination,
        viewTreeRoot,
        evaluateDemo,
        addImplicitAnswers,
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
      vscode.commands.registerCommand("delphyne.addImplicitAnswers", (add) => {
        add(); // The action to perform is passed as a closure argument.
      }),
    );
  }

  private evaluateDemoAction(
    element: DemoElement,
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

  private viewTreeRoot(element: DemoElement): vscode.CodeAction | null {
    const feedback = this.demosManager.getFeedback(
      element.uri,
      element.demo,
    ) as StrategyDemoFeedback;
    if (element.kind !== "strategy_demo") {
      return null;
    }
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

  private viewTestDestination(element: DemoElement): vscode.CodeAction | null {
    if (element.kind !== "strategy_demo") {
      return null;
    }
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

  private addImplicitAnswers(element: DemoElement): vscode.CodeAction | null {
    const editor = vscode.window.activeTextEditor;
    if (!editor || element.kind !== "strategy_demo") {
      return null;
    }
    const feedback = this.demosManager.getFeedback(
      element.uri,
      element.demo,
    ) as StrategyDemoFeedback | null;
    const demo = this.demosManager.getDemonstration(
      element.uri,
      element.demo,
    ) as StrategyDemo | null;
    if (!demo || !feedback || feedback.implicit_answers.length === 0) {
      return null;
    }
    const action = new vscode.CodeAction(
      "Add Implicit Answers",
      vscode.CodeActionKind.Empty,
    );
    action.command = {
      command: "delphyne.addImplicitAnswers",
      title: "Add Implicit Answers",
      arguments: [
        () => {
          const queries = feedback.implicit_answers.map((ia) => ({
            name: ia.query_name,
            args: ia.query_args,
            answer: ia.answer,
          }));
          addQueries(queries, demo, editor);
        },
      ],
    };
    return action;
  }

  private answerQuery(
    element: DemoElement,
    promptOnly: boolean,
  ): vscode.CodeAction | null {
    const query = queryOfDemoElement(element);
    if (!query) {
      return null;
    }
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
  element: DemoElement,
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
    await evaluateDemo(element, exe, server, demosManager);
  }
}

function openPointedTree(pointedTree: PointedTree, treeView: TreeView) {
  treeView.setPointedTree(pointedTree);
}
