open Why3
open Python_libgen

let set_debug_mode b =
  let modify = if b then Debug.set_flag else Debug.unset_flag in
  modify (Debug.lookup_flag "print_locs") ;
  modify (Debug.lookup_flag "print_attributes")

(* Internal errors corresponding to bugs in the why3py library. *)
exception Fatal of string

(* User errors *)
exception Error of string

let try_or_fail f = try f () with e -> raise (Error (Printexc.to_string e))

let throws_user_error f =
  try
    ignore (f ()) ;
    false
  with Error _ -> true

type 'a result = Answer of 'a | Error of string
[@@deriving python, python_export_type, show]

let wrap f = try Answer (f ()) with Error msg -> Error msg

let wrap_py to_py f () = (python_of_result to_py) (wrap f)

let input_filename = ""

type loc = int * int * int * int [@@deriving python, python_export_type, show]
(* bl, bc, el, ec *)

let convert_loc loc =
  let f, bl, bc, el, ec = Loc.get loc in
  if f = input_filename then Some (bl, bc, el, ec) else None

exception External_loc

let convert_loc_exn loc =
  match convert_loc loc with Some loc -> loc | None -> raise External_loc
