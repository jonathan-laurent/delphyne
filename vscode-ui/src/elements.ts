//////
/// Elements
//////

// What can be done on elements:
// - Track their liveness
// - Jump to them
// - Determine if one is under the cursor

import * as vscode from "vscode";

export type SerializedDemo = string;
export type SerializedCommand = string;

// Points to a demo or one of its specific components (test, query, answer...)
export interface DemoElement {
  kind: "demo";
  uri: vscode.Uri;
  demo: SerializedDemo;
  specific: TestElement | QueryElement | null;
}

// A specific test within a demonstration
export interface TestElement {
  kind: "test";
  test_index: number;
}

// A query within a demonstration
export interface QueryElement {
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

export type Element = DemoElement | CommandElement;
