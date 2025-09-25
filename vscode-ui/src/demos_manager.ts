//////
/// Managing demonstration files and the associated server feedback
//////

import * as vscode from "vscode";
import { Demo, DemoFile, QueryDemo } from "./stubs/demos";
import { StrategyDemo } from "./stubs/demos";
import { log } from "./logging";
import {
  DemoFeedback,
  Diagnostic,
  QueryDemoFeedback,
  StrategyDemoFeedback,
} from "./stubs/feedback";
import { parseYamlWithLocInfo, serializeWithoutLocInfo } from "./yaml_utils";
import { Ajv } from "ajv";
import { isStrategyDemo } from "./common";
import {
  TestElement,
  StrategyQueryElement,
  SerializedDemo,
  DemoElement,
} from "./elements";

const DEMO_SCHEMA_PATH = "../resources/demo-schema.json";

class DemoFileManager {
  constructor(private diagnosticCollection: vscode.DiagnosticCollection) {}
  private feedback = new Map<SerializedDemo, DemoFeedback>();
  private parsedDemoFile: DemoFile | null = null;

  public update(document: vscode.TextDocument) {
    this.parsedDemoFile = null;
    const parsed = parseYamlWithLocInfo(document.getText());
    if (isValidDemoFile(parsed)) {
      // Discard all feedback that is not relevant anymore from the cache
      const newFeedback = new Map<SerializedDemo, DemoFeedback>();
      for (const demo of parsed) {
        const serialized = serializeWithoutLocInfo(demo);
        if (this.feedback.has(serialized)) {
          newFeedback.set(serialized, this.feedback.get(serialized)!);
        }
      }
      this.feedback = newFeedback;
      this.parsedDemoFile = parsed;
    }
    this.updateDiagnostics(document.uri);
  }

  public receiveFeedback(
    uri: vscode.Uri,
    demo: SerializedDemo,
    feedback: DemoFeedback,
  ) {
    this.feedback.set(demo, feedback);
    this.updateDiagnostics(uri);
  }

  public getFeedback(demo: SerializedDemo): DemoFeedback | null {
    return this.feedback.get(demo) ?? null;
  }

  public getParsedDemoFile(): DemoFile | null {
    return this.parsedDemoFile;
  }

  public getElementAt(
    uri: vscode.Uri,
    cursor: vscode.Position,
  ): DemoElement | null {
    // Note that the URI is only necessary to annotate the resulting elements
    if (!this.parsedDemoFile) {
      return null;
    }
    return getElementAt(uri, this.parsedDemoFile, cursor);
  }

  public getAllDemoElements(uri: vscode.Uri): DemoElement[] | null {
    if (!this.parsedDemoFile) {
      return null;
    }
    return this.parsedDemoFile.map((demo) => {
      if (isStrategyDemo(demo)) {
        return {
          kind: "strategy_demo",
          uri: uri,
          demo: serializeWithoutLocInfo(demo),
          specific: null,
        };
      } else {
        return {
          kind: "standalone_query",
          uri: uri,
          demo: serializeWithoutLocInfo(demo),
          answer_index: null,
        };
      }
    });
  }

  public getDemonstration(demo: SerializedDemo): Demo | null {
    if (!this.parsedDemoFile) {
      return null;
    }
    for (const demonstration of this.parsedDemoFile) {
      if (serializeWithoutLocInfo(demonstration) === demo) {
        return demonstration;
      }
    }
    return null;
  }

  private updateDiagnostics(uri: vscode.Uri) {
    log.trace("Updating diagnostics for", uri.toString());
    let diagnostics: vscode.Diagnostic[] = [];
    if (!this.parsedDemoFile) {
      // If the file is invalid, we do not update anything
      return;
    }
    // Otherwise, we recompute all feedback
    for (const demonstration of this.parsedDemoFile) {
      const key = serializeWithoutLocInfo(demonstration);
      const feedback = this.feedback.get(key);
      if (!feedback) {
        continue;
      }
      diagnostics = diagnostics.concat(
        computeDiagnostics(demonstration, feedback),
      );
    }
    this.diagnosticCollection.set(uri, diagnostics);
  }
}

export function isValidDemoFile(file: unknown): file is DemoFile {
  const ajv = new Ajv();
  const schema = require(DEMO_SCHEMA_PATH);
  const valid = ajv.validate(schema, file);
  return valid;
}

//////
/// Diagnostic computation
//////

function makeDiagnostic([kind, message]: Diagnostic, loc: vscode.Range) {
  let severity =
    kind === "error"
      ? vscode.DiagnosticSeverity.Error
      : kind === "warning"
        ? vscode.DiagnosticSeverity.Warning
        : vscode.DiagnosticSeverity.Information;
  return new vscode.Diagnostic(loc, message, severity);
}

function computeDiagnostics(
  demonstration: Demo,
  feedback: DemoFeedback,
): vscode.Diagnostic[] {
  if (isStrategyDemo(demonstration)) {
    return computeStrategyDemoDiagnostics(
      demonstration,
      feedback as StrategyDemoFeedback,
    );
  } else {
    return computeQueryDemoDiagnostics(
      demonstration,
      feedback as QueryDemoFeedback,
    );
  }
}

function computeQueryDemoDiagnostics(
  demonstration: QueryDemo,
  feedback: QueryDemoFeedback,
): vscode.Diagnostic[] {
  let diagnostics: vscode.Diagnostic[] = [];
  for (let d of feedback.diagnostics) {
    diagnostics.push(makeDiagnostic(d, demonstration.__loc__query));
  }
  for (let [i, d] of feedback.answer_diagnostics) {
    diagnostics.push(makeDiagnostic(d, demonstration.__loc_items__answers[i]));
  }
  if (
    feedback.diagnostics.length === 0 &&
    feedback.answer_diagnostics.length === 0
  ) {
    diagnostics.push(
      makeDiagnostic(
        ["info", "Answers parsed successfully."],
        demonstration.__loc__query,
      ),
    );
  }
  return diagnostics;
}

function computeStrategyDemoDiagnostics(
  demonstration: StrategyDemo, // Annotated with location information
  feedback: StrategyDemoFeedback,
): vscode.Diagnostic[] {
  let diagnostics: vscode.Diagnostic[] = [];
  for (let d of feedback.global_diagnostics) {
    diagnostics.push(makeDiagnostic(d, demonstration.__loc__strategy));
  }
  for (let [i, d] of feedback.query_diagnostics) {
    diagnostics.push(makeDiagnostic(d, demonstration.__loc_items__queries[i]));
  }
  for (let [[i, j], d] of feedback.answer_diagnostics) {
    diagnostics.push(
      makeDiagnostic(d, demonstration.queries[i].__loc_items__answers[j]),
    );
  }
  for (let [i, f] of feedback.test_feedback.entries()) {
    let loc = demonstration.__loc_items__tests[i];
    for (let d of f.diagnostics) {
      diagnostics.push(makeDiagnostic(d, loc));
    }
    if (f.node_id !== null && f.diagnostics.length === 0) {
      diagnostics.push(makeDiagnostic(["info", "Test passed."], loc));
    }
  }
  for (const [category, answers] of Object.entries(feedback.implicit_answers)) {
    const msg = `Implicit answers were used: ${category} (${answers.length}).`;
    diagnostics.push(
      makeDiagnostic(["info", msg], demonstration.__loc__strategy),
    );
  }
  return diagnostics;
}

//////
/// Finding the element at the current cursor position
//////

// TODO: we do a lot of expensive serialization of demos

export function getElementAt(
  uri: vscode.Uri,
  demoFile: DemoFile,
  cursor: vscode.Position,
): DemoElement | null {
  for (const demo of demoFile) {
    const loc = demo.__loc;
    if (loc.contains(cursor)) {
      if (isStrategyDemo(demo)) {
        return {
          kind: "strategy_demo",
          uri: uri,
          demo: serializeWithoutLocInfo(demo),
          specific: getSpecificElementAt(demo, cursor),
        };
      } else {
        return {
          kind: "standalone_query",
          uri: uri,
          demo: serializeWithoutLocInfo(demo),
          answer_index: getAnswerIndex(demo, cursor),
        };
      }
    }
  }
  return null;
}

function getSpecificElementAt(
  demo: StrategyDemo,
  cursor: vscode.Position,
): TestElement | StrategyQueryElement | null {
  const testIndex = demo.__loc_items__tests.findIndex((loc) =>
    loc.contains(cursor),
  );
  if (testIndex >= 0) {
    return { kind: "test", test_index: testIndex };
  }
  const queryIndex = demo.__loc_items__queries.findIndex((loc) =>
    loc.contains(cursor),
  );
  if (queryIndex >= 0) {
    return {
      kind: "query",
      query_index: queryIndex,
      answer_index: getAnswerIndex(demo.queries[queryIndex], cursor),
    };
  }
  return null;
}

function getAnswerIndex(
  query: QueryDemo,
  cursor: vscode.Position,
): number | null {
  const answerIndex = query.__loc_items__answers.findIndex((loc) =>
    loc.contains(cursor),
  );
  return answerIndex >= 0 ? answerIndex : null;
}

//////
/// Global manager
//////

export class DemosManager {
  private demoFiles: Map<vscode.Uri, DemoFileManager> = new Map();
  private updateCallbacks: (() => void)[] = [];
  constructor(private diagnosticCollection: vscode.DiagnosticCollection) {}

  // Register an update callback
  onUpdate(callback: () => void) {
    this.updateCallbacks.push(callback);
  }

  private fireUpdateEvents() {
    for (const callback of this.updateCallbacks) {
      callback();
    }
  }

  onDidOpenTextDocument(document: vscode.TextDocument) {
    this.demoFiles.set(
      document.uri,
      new DemoFileManager(this.diagnosticCollection),
    );
    this.onDidChangeTextDocument(document);
  }

  onDidCloseTextDocument(document: vscode.TextDocument) {
    this.demoFiles.delete(document.uri);
    this.fireUpdateEvents();
  }

  onDidChangeTextDocument(document: vscode.TextDocument) {
    const manager = this.demoFiles.get(document.uri);
    if (!manager) {
      throw new Error(`No manager found for ${document.uri}.`);
    }
    manager.update(document);
    this.fireUpdateEvents();
  }

  receiveFeedback(
    uri: vscode.Uri,
    demo: SerializedDemo,
    feedback: DemoFeedback,
  ) {
    this.demoFiles.get(uri)?.receiveFeedback(uri, demo, feedback);
    this.fireUpdateEvents();
  }

  isAlive(origin: DemoElement): boolean {
    return this.getFeedback(origin.uri, origin.demo) !== null;
  }

  getFeedback(uri: vscode.Uri, demo: SerializedDemo): DemoFeedback | null {
    return this.demoFiles.get(uri)?.getFeedback(demo) ?? null;
  }

  getElementAt(uri: vscode.Uri, cursor: vscode.Position): DemoElement | null {
    return this.demoFiles.get(uri)?.getElementAt(uri, cursor) ?? null;
  }

  getDemonstration(uri: vscode.Uri, demo: SerializedDemo): Demo | null {
    return this.demoFiles.get(uri)?.getDemonstration(demo) ?? null;
  }

  getAllDemoElements(uri: vscode.Uri): DemoElement[] | null {
    return this.demoFiles.get(uri)?.getAllDemoElements(uri) ?? null;
  }

  getParsedDemoFile(uri: vscode.Uri): DemoFile | null {
    return this.demoFiles.get(uri)?.getParsedDemoFile() ?? null;
  }
}
