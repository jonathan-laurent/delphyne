let simple_loop =
  {|
    use int.Int

    let loop ()
      diverges
      ensures { result >= 0 }
    =
      let ref x = 0 in
      while x < 10 do
        invariant x_nonneg { x >= 0 }
        x <- x + 1
      done;
      x
    |}
