(* Right now, we only allow inserting new invariants. If we have a list,  *)

open Why3
open Api_common
open Python_libgen

exception Mismatch

type update = Added_invariant of loc
[@@deriving python, python_export_type, show]

type diff = Updates of update list | Mismatch of loc option
[@@deriving python, python_export_type, show]

class diff_visitor =
  object
    inherit [_] Ptree_derived.iter2

    method! visit_invariant (env : update Queue.t) invs invs' =
      let push_inv i =
        Queue.push (Added_invariant (convert_loc_exn i.Ptree.term_loc)) env
      in
      let rec aux invs invs' =
        match (invs, invs') with
        | [], invs' ->
            List.iter push_inv invs'
        | _, [] ->
            (* One cannot remove invariants *)
            raise VisitorsRuntime.StructuralMismatch
        | i :: invs, i' :: invs' ->
            if Ptree_utils.equal_term i i' then aux invs invs'
            else (
              push_inv i' ;
              aux (i :: invs) invs' )
      in
      aux invs invs'
  end

let diff mlw mlw' =
  let ast, ast' = (Ptree_utils.parse_mlw mlw, Ptree_utils.parse_mlw mlw') in
  let updates = Queue.create () in
  let compatible =
    VisitorsRuntime.wrap2 ((new diff_visitor)#visit_mlw_file updates) ast ast'
  in
  if compatible then Updates (Utils.list_of_queue updates) else Mismatch None
