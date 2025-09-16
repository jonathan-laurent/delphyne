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

export interface NestedTree {
  kind: "nested";
  strategy: string;
  args: Record<string, ValueRepr>;
  tags: string[];
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
  args: Record<string, unknown>;
  tags: string[];
  answers: Answer[];
}

export type NodeProperty = Data | NestedTree | Query;

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
  | ["nested", TraceNodeId, TraceNodePropertyId];

export interface Node {
  kind: string;
  success_value: ValueRepr | null;
  summary_message: string | null;
  leaf_node: boolean;
  label: string | null;
  tags: string[];
  properties: [Reference, NodeProperty][];
  actions: Action[];
  origin: NodeOrigin;
}

export interface Trace {
  nodes: Record<TraceNodeId, Node>;
}

export type DemoQueryId = number;

export type DemoAnswerId = [number, number];

export interface TestFeedback {
  diagnostics: Diagnostic[];
  node_id: TraceNodeId | null;
}

export interface ImplicitAnswer {
  query_name: string;
  query_args: Record<string, unknown>;
  answer: string;
}

export type ImplicitAnswerCategory = "computations" | "fetched" | string;

export interface StrategyDemoFeedback {
  kind: "strategy";
  trace: Trace;
  answer_refs: Record<TraceAnswerId, DemoAnswerId>;
  saved_nodes: Record<string, TraceNodeId>;
  test_feedback: TestFeedback[];
  global_diagnostics: Diagnostic[];
  query_diagnostics: [DemoQueryId, Diagnostic][];
  answer_diagnostics: [DemoAnswerId, Diagnostic][];
  implicit_answers: Record<ImplicitAnswerCategory, ImplicitAnswer[]>;
}

export interface QueryDemoFeedback {
  kind: "query";
  diagnostics: Diagnostic[];
  answer_diagnostics: [number, Diagnostic][];
}

export type DemoFeedback = StrategyDemoFeedback | QueryDemoFeedback;
