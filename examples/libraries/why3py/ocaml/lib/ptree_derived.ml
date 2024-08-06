open Why3
open Python_libgen

module Derived = struct
  module Opaque = struct
    [@@@ocamlformat "module-item-spacing=preserve"]

    type position = Why3.Loc.position
    let position_of_python py =
      let f, bl, bc, el, ec = [%of_python: string * int * int * int * int] py in
      Loc.user_position f bl bc el ec
    let python_of_position pos =
      [%python_of: string * int * int * int * int] (Loc.get pos)
    let () =
      let open Python_libgen.Repr in
      let string = App (String, []) and int = App (Int, []) in
      register_python_type
        { type_name= "Position"
        ; type_vars= []
        ; definition= Alias (Tuple [string; int; int; int; int]) }

    type big_int = BigInt.t
    let equal_big_int = BigInt.eq
    let big_int_of_python py = BigInt.of_string ([%of_python: string] py)
    let python_of_big_int x = [%python_of: string] (BigInt.to_string x)
    let () =
      register_python_type
        { type_name= "BigInt"
        ; type_vars= []
        ; definition= Alias (Tuple [App (String, [])]) }

    type attribute_ident = Why3.Ident.attribute
    let equal_attribute_ident a a' = Ident.attr_compare a a' = 0
    let python_of_attribute_ident a = [%python_of: string] a.Ident.attr_string
    let attribute_ident_of_python py =
      Ident.create_attribute ([%of_python: string] py)
    let () =
      register_python_type
        { type_name= "AttributeIdent"
        ; type_vars= []
        ; definition= Alias (Tuple [App (String, [])]) }

    type real_value = Why3.Number.real_value
    let equal_real_value rv rv' = Number.compare_real rv rv' = 0
    let real_value_of_python py =
      let s, pow2, pow5 = [%of_python: big_int * big_int * big_int] py in
      Number.real_value ~pow2 ~pow5 s
    let python_of_real_value rv =
      [%python_of: big_int * big_int * big_int]
        Number.(rv.rv_sig, rv.rv_pow2, rv.rv_pow5)
    let () =
      let open Python_libgen.Repr in
      let big_int = App (Custom "BigInt", []) in
      register_python_type
        { type_name= "RealValue"
        ; type_vars= []
        ; definition= Alias (Tuple [big_int; big_int; big_int]) }
  end

  type position = (Opaque.position[@opaque])

  and big_int = (Opaque.big_int[@opaque])

  and attribute_ident = (Opaque.attribute_ident[@opaque])

  and real_value = (Opaque.real_value[@opaque])

  and real_literal_kind = [%import: Why3.Number.real_literal_kind]

  and real_constant = [%import: Why3.Number.real_constant]

  and int_literal_kind = [%import: Why3.Number.int_literal_kind]

  and int_value = [%import: (Why3.Number.int_value[@with BigInt.t := big_int])]

  and int_constant = [%import: Why3.Number.int_constant]

  and constant =
    [%import:
      (Why3.Constant.constant
      [@with
        Number.real_constant := real_constant ;
        Number.int_constant := int_constant] )]

  and dbinop = [%import: Why3.Dterm.dbinop]

  and dquant = [%import: Why3.Dterm.dquant]

  and prop_kind = [%import: Why3.Decl.prop_kind]

  and ind_sign = [%import: Why3.Decl.ind_sign]

  and rs_kind = [%import: Why3.Expr.rs_kind]

  and assertion_kind = [%import: Why3.Expr.assertion_kind]

  and for_direction = [%import: Why3.Expr.for_direction]

  and mask = [%import: Why3.Ity.mask]

  (* opaque *)
  and attr =
    [%import:
      (Why3.Ptree.attr
      [@with
        Ident.attribute := attribute_ident ;
        Loc.position := position] )]

  and ident = [%import: (Why3.Ptree.ident[@with Loc.position := position])]

  and qualid = [%import: Why3.Ptree.qualid]

  and pty = [%import: Why3.Ptree.pty]

  and ghost = [%import: Why3.Ptree.ghost]

  and pattern = [%import: (Why3.Ptree.pattern[@with Loc.position := position])]

  and pat_desc = [%import: Why3.Ptree.pat_desc]

  and binder = [%import: (Why3.Ptree.binder[@with Loc.position := position])]

  and param = [%import: (Why3.Ptree.param[@with Loc.position := position])]

  and term = [%import: (Why3.Ptree.term[@with Loc.position := position])]

  and term_desc =
    [%import:
      (Why3.Ptree.term_desc
      [@with
        Dterm.dbinop := dbinop ;
        Dterm.dquant := dquant ;
        Constant.constant := constant] )]

  and invariant = [%import: Why3.Ptree.invariant]

  and variant = [%import: Why3.Ptree.variant]

  and pre = [%import: Why3.Ptree.pre]

  and post = [%import: (Why3.Ptree.post[@with Loc.position := position])]

  and xpost = [%import: (Why3.Ptree.xpost[@with Loc.position := position])]

  and spec = [%import: Why3.Ptree.spec]

  and expr = [%import: (Why3.Ptree.expr[@with Loc.position := position])]

  and expr_desc =
    [%import:
      (Why3.Ptree.expr_desc
      [@with
        Loc.position := position ;
        Expr.rs_kind := rs_kind ;
        Expr.assertion_kind := assertion_kind ;
        Expr.for_direction := for_direction ;
        Constant.constant := constant ;
        Ity.mask := mask] )]

  and reg_branch = [%import: Why3.Ptree.reg_branch]

  and exn_branch = [%import: Why3.Ptree.exn_branch]

  and fundef =
    [%import:
      (Why3.Ptree.fundef
      [@with
        Loc.position := position ;
        Expr.rs_kind := rs_kind ;
        Ity.mask := mask] )]

  and field = [%import: (Why3.Ptree.field[@with Loc.position := position])]

  and type_def =
    [%import:
      (Why3.Ptree.type_def
      [@with
        Loc.position := position ;
        BigInt.t := big_int] )]

  and visibility = [%import: Why3.Ptree.visibility]

  and type_decl =
    [%import: (Why3.Ptree.type_decl[@with Loc.position := position])]

  and logic_decl =
    [%import: (Why3.Ptree.logic_decl[@with Loc.position := position])]

  and ind_decl =
    [%import: (Why3.Ptree.ind_decl[@with Loc.position := position])]

  and metarg = [%import: Why3.Ptree.metarg]

  and clone_subst =
    [%import: (Why3.Ptree.clone_subst[@with Decl.prop_kind := prop_kind])]

  and decl =
    [%import:
      (Why3.Ptree.decl
      [@with
        Loc.position := position ;
        Decl.prop_kind := prop_kind ;
        Decl.ind_sign := ind_sign ;
        Expr.rs_kind := rs_kind ;
        Ity.mask := mask] )]

  and mlw_file = [%import: Why3.Ptree.mlw_file]
  [@@deriving
    python
    , python_export_type
    , visitors {variety= "iter"}
    , visitors {variety= "map"}
    , visitors {variety= "iter2"}]
end

include Derived

let check_eq_with f x x' =
  if not (f x x') then raise VisitorsRuntime.StructuralMismatch

class ['a] iter2 =
  object (_ : 'a)
    inherit [_] Derived.iter2

    method! visit_position _env _p _p' = () (* disregard location annots *)

    method! visit_attribute_ident _env =
      check_eq_with [%eq: Opaque.attribute_ident]

    method! visit_real_value _env = check_eq_with [%eq: Opaque.real_value]

    method! visit_big_int _env = check_eq_with [%eq: Opaque.big_int]
  end
