open Python_libgen
open Api_common
open Why3

type color = Premise | Goal [@@deriving python, python_export_type, show]

type obligation =
  { name: string
  ; task: string
  ; goal_loc: loc option
  ; locs: (loc * color) list
  ; proved: bool
  ; prover_answer: string
  ; prover_steps: int }
[@@deriving python, python_export_type, show]

(* Global initialization *)

let config = Whyconf.init_config None

let config_main = Whyconf.get_main config

let env = Env.create_env (Whyconf.loadpath config_main)

let split_trans = Trans.lookup_transform_l "split_vc" env

let alt_ergo =
  let fp = Whyconf.parse_filter_prover "Alt-Ergo" in
  let provers = Whyconf.filter_provers config fp in
  if Whyconf.Mprover.is_empty provers then
    raise (Fatal (Fmt.str "Prover Alt-Ergo not installed or not configured"))
  else snd (Whyconf.Mprover.max_binding provers)

let alt_ergo_driver =
  try Driver.load_driver_for_prover config_main env alt_ergo
  with e ->
    raise
      (Fatal
         (Fmt.str "Failed to load driver for alt-ergo: %a."
            Exn_printer.exn_printer e ) )

(* Prove everything *)

let compute_tasks ast =
  let mods =
    try_or_fail (fun () -> Typing.type_mlw_file env [] input_filename ast)
  in
  let tasks =
    Wstdlib.Mstr.fold
      (fun _ m acc ->
        List.rev_append (Task.split_theory m.Pmodule.mod_theory None None) acc
        )
      mods []
  in
  List.rev tasks

let prove_task ~max_steps ~max_time_in_secs t =
  let open Call_provers in
  let limits = {empty_limits with limit_steps= max_steps; limit_time = max_time_in_secs} in
  let res =
    wait_on_call
      (Driver.prove_task ~limits ~config:config_main
         ~command:alt_ergo.Whyconf.command alt_ergo_driver t )
  in
  let proved = res.pr_answer = Valid in
  let answer = Fmt.str "%a" (print_prover_result ~json:false) res in
  let num_steps = res.pr_steps in
  (proved, answer, num_steps)

let split t = Trans.apply split_trans t

let subst_all t = Trans.apply Subst.subst_all t

let explanation t =
  let _, expl, _ = Termcode.goal_expl_task ~root:false t in
  expl

(* Locations *)

let get_locations (task : Task.task) =
  (* Inspired from itp_server.ml *)
  let list = ref [] in
  let goal_loc = ref None in
  let color_loc ~color ~loc =
    convert_loc loc |> Option.iter (fun loc -> list := (loc, color) :: !list)
  in
  let get_goal_loc ~loc pr_loc =
    (* We get the task loc in priority. If it is not there, we take the pr
       ident loc. No loc can happen in completely ghost function. *)
    let loc = Opt.fold (fun _ x -> Some x) pr_loc loc in
    match loc with Some loc -> goal_loc := convert_loc loc | _ -> ()
  in
  let rec color_locs ~color formula =
    Option.iter (fun loc -> color_loc ~color ~loc) formula.Term.t_loc ;
    Term.t_iter (fun subf -> color_locs ~color subf) formula
  in
  let rec color_t_locs ~premise f =
    match f.Term.t_node with
    | Term.Tbinop (Term.Timplies, f1, f2) when not premise ->
        color_t_locs ~premise:true f1 ;
        color_t_locs ~premise:false f2
    | Term.Tbinop (Term.Tand, f1, f2) when premise ->
        color_t_locs ~premise f1 ; color_t_locs ~premise f2
    | Term.Tlet (_, fb) ->
        let _, f1 = Term.t_open_bound fb in
        color_t_locs ~premise f1
    | Term.Tquant (Term.Tforall, fq) when not premise ->
        let _, _, f1 = Term.t_open_quant fq in
        color_t_locs ~premise f1
    | Term.Tnot f1 when premise && f.Term.t_loc = None ->
        (* originally: Negated_premise_color *)
        color_locs ~color:Premise f1
    | _ when premise ->
        color_locs ~color:Premise f
    | _ ->
        color_locs ~color:Goal f
  in
  let color_goal = function
    | None ->
        raise (Fatal "No loc on goal.")
    | Some loc ->
        color_loc ~color:Goal ~loc
  in
  let goal_id : Ident.ident = (Task.task_goal task).Decl.pr_name in
  color_goal goal_id.Ident.id_loc ;
  (* Tasks have a linked list structure *)
  let rec scan = function
    | Some
        { Task.task_prev= prev
        ; Task.task_decl=
            { Theory.td_node= Theory.Decl {Decl.d_node= Decl.Dprop (k, pr, f); _}
            ; _ }
        ; _ } ->
        ( match k with
        | Decl.Pgoal ->
            get_goal_loc ~loc:f.Term.t_loc pr.Decl.pr_name.Ident.id_loc ;
            color_t_locs ~premise:false f
        | Decl.Paxiom ->
            color_t_locs ~premise:true f
        | _ ->
            assert false ) ;
        scan prev
    | Some {Task.task_prev= prev; _} ->
        scan prev
    | _ ->
        ()
  in
  scan task ;
  (!goal_loc, List.rev !list)

(* Main function *)

let prove ~max_steps ~max_time_in_secs src =
  let ast = Ptree_utils.parse_mlw src in
  let tasks = compute_tasks ast in
  let process task =
    let top_expl = explanation task in
    let subs = split task in
    subs
    |> List.map (fun sub ->
           let expl = explanation sub in
           let name = Fmt.str "%s (%s)" top_expl expl in
           let goal_loc, locs = get_locations sub in
           (* The locations are more meaningful BEFORE subst_all is applied *)
           let sub = subst_all sub in
           let task = Fmt.str "%a" Pretty.print_sequent sub in
           let proved, prover_answer, prover_steps =
             prove_task ~max_steps sub ~max_time_in_secs
           in
           {name; task; goal_loc; locs; proved; prover_answer; prover_steps} )
  in
  List.concat_map process tasks

(* Tests *)

let%test_unit "parse error" =
  assert (throws_user_error (fun () -> prove {| let x: int = true |}))

let%test_unit "success" =
  assert (not (throws_user_error (fun () -> prove Examples.simple_loop)))
