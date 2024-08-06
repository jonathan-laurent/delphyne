open Api_common

(** Parsing utilities  *)

let parse_mlw src =
  let lexbuf = Lexing.from_string ~with_positions:true src in
  Lexing.set_filename lexbuf input_filename ;
  let mlw_file =
    try Why3.Lexer.parse_mlw_file lexbuf
    with e -> raise (Error (Printexc.to_string e))
  in
  mlw_file

(** Equality checking *)

let equal_mlw_file =
  VisitorsRuntime.wrap2 ((new Ptree_derived.iter2)#visit_mlw_file ())

let equal_term = VisitorsRuntime.wrap2 ((new Ptree_derived.iter2)#visit_term ())

let%test_unit "equal" =
  let eq s s' = equal_mlw_file (parse_mlw s) (parse_mlw s') in
  let sloop = Examples.simple_loop in
  assert (eq sloop sloop) ;
  assert (eq sloop (String.concat "\n" [" "; sloop])) ;
  assert (not (eq {| let x = 4 |} {| let y = 4 |}))
