import MiniF2F.Minif2fImport
import Aesop

-- Inductive definitions of odd and even predicates
inductive MyEven : ℕ → Prop where
  | zero : MyEven 0
  | succ_succ : ∀ n, MyEven n → MyEven (n + 2)

inductive MyOdd : ℕ → Prop where
  | one : MyOdd 1
  | succ_succ : ∀ n, MyOdd n → MyOdd (n + 2)

-- Let's first check out whether the tactics we want are available

example (x: ℝ): x^2 + 1 ≥ 0 := by
  positivity

example (n: ℕ) (h₁: n < 6): (n - 2)^2 <= 10 := by
  interval_cases n <;> norm_num

-- Let's see how complicated things can be evaluated
example: ∑ i ∈ Finset.range 1001, i = (1000)*(1001)/2 := by
  -- decide -- Maximum recursion depth reached
  -- set_option maxRecDepth 1 in decide -- Why does this work?
  -- set_option maxRecDepth 10 in decide -- Doesn't work
  set_option maxRecDepth 10_000 in decide -- Works

theorem simple_proof: ∀ x: ℝ, (x - 1)^2 + 2 ≥ 2 := by
  intro x
  have: (x - 1)^2 ≥ 0 := by exact sq_nonneg (x - 1)  -- apply?
  -- try omega ; try linarith
  simpa

example (a b : ℕ) (h₁ : a < b) (h₂ : b ≤ c) : a + d ≤ c + d := by
  grw [h₁, h₂]

--
-- Testing Grind
--

example (α: Type) (f g : α → α) (h: ∀ x, f (f x) = g x) (x: α): (f (g x) = g (f x)) := by
  grind

-- From Zulip
example {P : Nat → Prop} {m : Nat} (hm₀ : 0 < m) (hm₂ : m ≤ 2) (h : P m) : P 1 ∨ P 2 := by
  -- grind -- does not work
  grind [cases Nat]

-- Grind does pretty well with arithmetic
theorem amc12a_2017_p2 (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : x + y = 4 * (x * y)) :
  1 / x + 1 / y = 4 := by
  grind

example: MyEven 4 := by grind [MyEven]

example (x: ℕ) (h: 2 * x > 0): ∃ y, y = x + 1 :=
  by grind

def f: ℕ → ℕ
  | 0 => 1
  | _ + 1 => 2  -- Better than _ => ... for grind

example: ∀ x: ℕ, f x < 3 := by
  -- set_option trace.grind true in
  grind [cases Nat, f]

example (f g : ℕ → ℕ) (h: ∀ x, g (f x) = x) (a b : ℕ)
  (h₁: f b = a) (h₂: f c = a): b = c := by grind

example: ∃ x, f x = 2 := by use 2 ; grind [f]

-- example: ∃ x, f x = 2 := by grind [f] -- No

example (P: Prop): P ∨ ¬P := by grind

example (P: Prop): P ∨ ¬P := by grind

example: ∀ P Q : ℕ → Prop, (∀ x, P x → Q x) → (∃ x, P x) -> ∃ x, Q x := by
  grind

example:
  ∀ n: ℕ,
  ∑ i ∈ Finset.range n, (f i + 1) = (∑ i ∈ Finset.range n, f i) + n := by
  -- grind
  intro n ;
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range]
  ring

--
-- Solving some simple examples
--

-- Using grind
example (x : ℕ) (h₁ : x > 0) : x >= 1 := by
  grind

-- For auto
theorem mathd_numbertheory_127 : (∑ k ∈ Finset.range 101, 2 ^ k) % 7 = 3 := by
  decide

-- calculation, forward + show loogle
theorem mathd_algebra_22 : Real.logb (5 ^ 2) (5 ^ 4) = 2 := by
  calc
    Real.logb (5 ^ 2) (5 ^ 4)
    _ = Real.logb (5 ^ 2) ((5 ^ 2) ^ 2) := by rw [show ((5: ℝ) ^ 2) ^ 2 = 5 ^ 4 by ring]
    _ = 2 := by simp [Real.logb_pow]

#loogle Real.logb, "pow"

-- one_liner
theorem amc12b_2003_p9 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = a * x + b) (h₁ : f 6 - f 2 = 12) :
    f 12 - f 2 = 30 := by
  simp [h₀] at h₁ ⊢ ; linarith

-- one_liner
theorem mathd_algebra_192 (q e d : ℂ) (h₀ : q = 11 - 5 * Complex.I) (h₁ : e = 11 + 5 * Complex.I)
    (h₂ : d = 2 * Complex.I) : q * e * d = 292 * Complex.I := by
  rw [h₀, h₁, h₂] ; ring_nf ; simp ; ring_nf

-- one_liner
theorem mathd_algebra_433 (f : ℝ → ℝ) (h₀ : ∀ x, f x = 3 * Real.sqrt (2 * x - 7) - 8) : f 8 = 1 := by
  rw [h₀] ; norm_num

-- forward
theorem amc12_2001_p9 (f : ℝ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y) (h₁ : f 500 = 3) :
  f 600 = 5 / 2 := by
  have: (600: ℝ) = 500 * (6 / 5) := by norm_num
  have: f 600 = 5 / 2 := by rw [this, h₀, h₁] <;> norm_num
  assumption

-- Simple arithmetic
theorem amc12_2000_p11 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a * b = a - b) :
    a / b + b / a - a * b = 2 := by
  -- grind -- Works straight away
  field_simp [h₀.1, h₀.2]
  rw [show a^2 + b^2 = (a-b)^2 + 2*a*b by ring]
  rw [←h₁]
  ring

-- one_liner, brute_force
set_option linter.unusedVariables false
theorem mathd_numbertheory_780 (m x : ℤ) (h₀ : 0 ≤ x) (h₁ : 10 ≤ m ∧ m ≤ 99) (h₂ : 6 * x % m = 1)
  (h₃ : (x - 6 ^ 2) % m = 0) : m = 43 := by
  rcases h₁ ; interval_cases m <;> omega

-- one_liner, brute_force
theorem mathd_numbertheory_109 (v : ℕ → ℕ) (h₀ : ∀ n, v n = 2 * n - 1) :
  (∑ k ∈ Finset.Icc 1 100, v k) % 7 = 4 := by
  simp [h₀] ; rfl

-- one_liner
theorem amc12a_2017_p2_bis (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : x + y = 4 * (x * y)) :
  1 / x + 1 / y = 4 := by
  field_simp ; linarith

-- forward
theorem numbertheory_2dvd4expn (n : ℕ) (h₀ : n ≠ 0) : 2 ∣ 4 ^ n := by
  have: n = (n - 1) + 1 := by omega
  -- Careful not to rewrite the wrong one...
  have: 4 ^ n = 2 * (2 * 4 ^ (n - 1)) := by nth_rw 1 [this] ; ring
  omega

-- forward
theorem amc12b_2003_p17 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : Real.log (x * y ^ 3) = 1)
  (h₂ : Real.log (x ^ 2 * y) = 1) : Real.log (x * y) = 3 / 5 := by
    have: x ≠ 0 := by linarith
    have: y ≠ 0 := by linarith
    have: y^3 ≠ 0 := by positivity
    have: x^2 ≠ 0 := by positivity
    have: Real.log x + 3 * Real.log y = 1 := by rw [Real.log_mul, Real.log_pow] at h₁ <;> assumption
    have: 2 * Real.log x + Real.log y = 1 := by rw [Real.log_mul, Real.log_pow] at h₂ <;> assumption
    have: Real.log x + Real.log y = 3 / 5 := by linarith
    rw [Real.log_mul] <;> assumption


--
-- Omega-ed stuff
--

theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
  norm_num

theorem mathd_numbertheory_136 (n : ℕ) (h₀ : 123 * n + 17 = 39500) : n = 321 := by
  omega

theorem mathd_numbertheory_284 (a b : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9)
  (h₁ : 10 * a + b = 2 * (a + b)) : 10 * a + b = 18 := by
  omega

theorem mathd_numbertheory_370 (n : ℕ) (h₀ : n % 7 = 3) : (2 * n + 1) % 7 = 0 := by
  omega


--
-- Difficult Problems
--

theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
  ((∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * (∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  simp [Finset.sum_Icc_succ_top, h₀]
  -- Hard to compute with complex numbers
  sorry

#loogle Finset.Icc, "succ"
#loogle ∑ _ ∈ Finset.Icc _ (_ + 1), _

theorem mathd_numbertheory_461 (n : ℕ)
  (h₀ : n = Finset.card (Finset.filter (fun x => Nat.gcd x 8 = 1) (Finset.Icc 1 7))) :
  3 ^ n % 8 = 1 := by
  simp [Finset.filter] at h₀
  sorry

theorem mathd_algebra_89 (b : ℝ) (h₀ : b ≠ 0) :
  (7 * b ^ 3) ^ 2 * (4 * b ^ 2) ^ (-(3 : ℤ)) = 49 / 64 := by
  simp [mul_zpow]
  sorry
  -- have: (4 * b ^ 2) ^ (-(3 : ℤ)) = 4^(-3: ℤ) * (b ^ (-6: ℤ)) := by rw [mul_zpow]
  -- sorry
  -- by
  -- calc
  --   (7 * b ^ 3) ^ 2 * (4 * b ^ 2) ^ (-(3 : ℤ))
  --   _ = (49 / 64) * (b ^ 2) ^ (-(3 : ℤ)) := by sorry
  --   _ = _ := by sorry
