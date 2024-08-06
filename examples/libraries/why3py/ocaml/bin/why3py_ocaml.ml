open Why3py
open Why3py.Api_common
open Python_libgen

let%python_export prove (src : string) : Prove.obligation list result =
  wrap (fun () -> Prove.prove src)

let%python_export diff (original : string) (modified : string) :
    Diff.diff result =
  wrap (fun () -> Diff.diff original modified)

let () = Driver.run ~generated_module:"core"
