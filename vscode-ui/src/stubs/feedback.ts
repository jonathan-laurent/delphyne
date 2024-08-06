export type DiagnosticType = "error" | "warning" | "info";

export type Diagnostic = [DiagnosticType, string];

export type TraceNodeId = number;

export type TraceAnswerId = number;

export type TraceActionId = number;

export type TraceNodePropertyId = number;

export interface ValueRepr {
  short: string;
  long: string | null;
  json_provided: boolean;
  json: unknown;
}

export interface Reference {
  with_ids: string;
  with_hints: string | null;
}

export interface Data {
  kind: "data";
  content: string;
}

export interface Subtree {
  kind: "subtree";
  strategy: string;
  args: { [key: string]: ValueRepr };
  node_id: TraceNodeId | null;
}

export interface Answer {
  id: TraceAnswerId;
  hint: [] | [string] | null;
  value: ValueRepr;
}

export interface Query {
  kind: "query";
  name: string;
  args: { [key: string]: unknown };
  answers: Answer[];
}

export type NodeProperty = Data | Subtree | Query;

export interface Action {
  ref: Reference;
  hints: string[] | null;
  related_success_nodes: TraceNodeId[];
  related_answers: TraceAnswerId[];
  value: ValueRepr;
  destination: TraceNodeId;
}

export type NodeOrigin =
  | "root"
  | ["child", TraceNodeId, TraceActionId]
  | ["sub", TraceNodeId, TraceNodePropertyId];

export interface Node {
  kind: string;
  success_value: ValueRepr | null;
  summary_message: string | null;
  leaf_node: boolean;
  label: string | null;
  properties: [Reference, NodeProperty][];
  actions: Action[];
  origin: NodeOrigin;
}

export interface Trace {
  nodes: { [key: number]: Node };
}

export type DemoQueryId = number;

export type DemoAnswerId = [number, number];

export interface TestFeedback {
  diagnostics: Diagnostic[];
  node_id: TraceNodeId | null;
}

export interface DemoFeedback {
  trace: Trace;
  answer_refs: { [key: number]: DemoAnswerId };
  saved_nodes: { [key: string]: TraceNodeId };
  test_feedback: TestFeedback[];
  global_diagnostics: Diagnostic[];
  query_diagnostics: [DemoQueryId, Diagnostic][];
  answer_diagnostics: [DemoAnswerId, Diagnostic][];
}
