//////
/// Elements
//////

// What can be done on elements:
// - Track their liveness
// - Jump to them
// - Determine if one is under the cursor

import * as vscode from "vscode";
import { QueryDemo, StrategyDemo } from "./stubs/demos";

export type SerializedDemo = string;
export type SerializedCommand = string;

// Points to a strategy demo or one of its specific components (test, query, answer...)
export interface StrategyDemoElement {
  kind: "strategy_demo";
  uri: vscode.Uri;
  demo: SerializedDemo;
  specific: TestElement | StrategyQueryElement | null;
}

// Points to a query demo or one of its specific answers.
export interface StandaloneQueryDemoElement {
  kind: "standalone_query";
  uri: vscode.Uri;
  demo: SerializedDemo;
  answer_index: number | null;
}

// A specific test within a demonstration
export interface TestElement {
  kind: "test";
  test_index: number;
}

// A query within a demonstration
export interface StrategyQueryElement {
  kind: "query";
  query_index: number;
  answer_index: number | null;
}

// Command elements
export interface CommandElement {
  kind: "command";
  uri: vscode.Uri;
  command: SerializedCommand;
}

export type DemoElement = StrategyDemoElement | StandaloneQueryDemoElement;

export type Element =
  | StrategyDemoElement
  | StandaloneQueryDemoElement
  | CommandElement;

export function queryOfDemoElement(element: DemoElement): QueryDemo | null {
  if (element.kind === "strategy_demo") {
    const specific = element.specific;
    if (!specific || specific.kind !== "query") {
      return null;
    }
    const demo = JSON.parse(element.demo) as StrategyDemo;
    return demo.queries[specific.query_index];
  } else {
    const demo = JSON.parse(element.demo) as QueryDemo;
    return demo;
  }
}
