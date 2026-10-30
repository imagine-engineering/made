use crate::numeric_input::{NumberInputConstraints, NumberInputError, evaluate_number_input};

fn original_real(expression: &str) -> f64 {
    graphite_math_parser::evaluate(expression)
        .expect("original parser accepts this expression")
        .expect("original evaluator accepts this expression")
        .as_real()
        .expect("original result is real")
}

#[test]
fn number_input_arithmetic_and_functions_use_original_expression_evaluation() {
    for (expression, expected) in [
        ("5+1*1+5", 11.),
        ("(5 + 3) * (2 + 6)", 64.),
        ("sqrt(16) + sqrt(9) * sqrt(4)", 10.),
        ("2sqrt(4)", 4.),
        ("sin(pi/2)", 1.),
        ("2^3", 8.),
        ("mod(-7, 3)", 2.),
    ] {
        let original = original_real(expression);
        assert!((original - expected).abs() <= 1e-10, "{expression}");
        let evaluated = evaluate_number_input(expression, NumberInputConstraints::default())
            .expect("same original expression is admitted");
        assert_eq!(evaluated.to_bits(), original.to_bits(), "{expression}");
    }
}

#[test]
fn number_input_scientific_notation_and_unicode_infinity_retain_source_values() {
    for (expression, expected) in [
        ("2e-1", 0.2),
        ("2E+1", 20.),
        ("2.5e3", 2500.),
        ("∞", f64::INFINITY),
        ("-∞", f64::NEG_INFINITY),
        ("1/0", f64::INFINITY),
    ] {
        assert_eq!(
            original_real(expression).to_bits(),
            expected.to_bits(),
            "{expression}"
        );
        assert_eq!(
            evaluate_number_input(expression, NumberInputConstraints::default())
                .expect("source real value")
                .to_bits(),
            expected.to_bits(),
            "{expression}"
        );
    }
    let bounded = NumberInputConstraints {
        minimum: Some(0.),
        maximum: Some(100.),
        is_integer: false,
    };
    assert_eq!(evaluate_number_input("∞", bounded), Ok(100.));
    assert_eq!(evaluate_number_input("-∞", bounded), Ok(0.));
}

#[test]
fn number_input_accepts_original_complex_expressions_only_after_real_reduction() {
    for (expression, expected) in [
        ("i * i", -1.),
        ("(1 + i) * (1 - i)", 2.),
        ("sqrt(-4) * sqrt(-4)", -4.),
    ] {
        assert_eq!(original_real(expression), expected, "{expression}");
        assert_eq!(
            evaluate_number_input(expression, NumberInputConstraints::default()),
            Ok(expected),
            "{expression}"
        );
    }
    for expression in ["i", "j", "sqrt(-1)", "I", "[1;2]"] {
        let original = graphite_math_parser::evaluate(expression)
            .expect("original expression parses")
            .expect("original non-real value evaluates");
        assert!(original.as_real().is_none(), "{expression}");
        assert_eq!(
            evaluate_number_input(expression, NumberInputConstraints::default()),
            Err(NumberInputError::NonReal),
            "{expression}"
        );
    }
}

#[test]
fn number_input_preserves_original_parse_errors_and_rejects_trailing_units_or_percent() {
    for expression in ["", "1.5.5", "2@", "sqrt(4)@", "(1", "50%", "5 % 3"] {
        let original = graphite_math_parser::evaluate(expression)
            .expect_err("original expression is malformed");
        let error = evaluate_number_input(expression, NumberInputConstraints::default())
            .expect_err("unchanged expression must fail parsing");
        assert_eq!(
            error,
            NumberInputError::Parse(original.to_string()),
            "{expression}"
        );
        assert_eq!(
            error.to_string(),
            format!("Invalid numeric expression: {original}")
        );
        if expression.contains('%') {
            assert!(original.to_string().contains("mod(a, b)"));
        }
    }
    let original = graphite_math_parser::evaluate("100px")
        .expect("source accepts implicit multiplication syntax")
        .expect_err("source has no variable binding for px");
    assert_eq!(
        evaluate_number_input("100px", NumberInputConstraints::default()),
        Err(NumberInputError::Evaluation(original.to_string()))
    );
    assert_eq!(
        evaluate_number_input("50", NumberInputConstraints::default()),
        Ok(50.)
    );
}

#[test]
fn number_input_indeterminate_forms_fail_before_clamping_or_integer_rounding() {
    let constraints = NumberInputConstraints {
        minimum: Some(0.),
        maximum: Some(100.),
        is_integer: true,
    };
    for expression in ["0/0", "inf - inf", "0 * inf", "sin(inf)", "mod(5, 0)"] {
        let original = graphite_math_parser::evaluate(expression)
            .expect("original indeterminate expression parses")
            .expect_err("original evaluator rejects indeterminate forms");
        let error = evaluate_number_input(expression, constraints)
            .expect_err("constraints cannot rescue an evaluator failure");
        assert_eq!(
            error,
            NumberInputError::Evaluation(original.to_string()),
            "{expression}"
        );
        assert_eq!(
            error.to_string(),
            format!("Numeric evaluation failed: {original}")
        );
    }
    assert_eq!(
        evaluate_number_input("sqrt(-1)", constraints),
        Err(NumberInputError::NonReal)
    );
    assert!(matches!(
        evaluate_number_input("25%", constraints),
        Err(NumberInputError::Parse(_))
    ));
}

#[test]
fn number_input_applies_minimum_then_maximum_then_round_without_reclamping() {
    for (expression, constraints, expected) in [
        (
            "-1",
            NumberInputConstraints {
                minimum: Some(0.),
                maximum: Some(100.),
                is_integer: false,
            },
            0.,
        ),
        (
            "150",
            NumberInputConstraints {
                minimum: Some(0.),
                maximum: Some(100.),
                is_integer: false,
            },
            100.,
        ),
        (
            "0",
            NumberInputConstraints {
                minimum: Some(10.),
                maximum: Some(5.),
                is_integer: false,
            },
            5.,
        ),
        (
            "0",
            NumberInputConstraints {
                minimum: Some(0.6),
                maximum: Some(1.4),
                is_integer: true,
            },
            1.,
        ),
        (
            "10",
            NumberInputConstraints {
                minimum: Some(1.2),
                maximum: Some(1.4),
                is_integer: true,
            },
            1.,
        ),
        (
            "-99",
            NumberInputConstraints {
                minimum: Some(1.6),
                maximum: Some(1.8),
                is_integer: true,
            },
            2.,
        ),
        (
            "-2.5",
            NumberInputConstraints {
                minimum: None,
                maximum: None,
                is_integer: true,
            },
            -3.,
        ),
        (
            "2.5",
            NumberInputConstraints {
                minimum: None,
                maximum: None,
                is_integer: true,
            },
            3.,
        ),
    ] {
        assert_eq!(
            evaluate_number_input(expression, constraints),
            Ok(expected),
            "{expression}, {constraints:?}"
        );
    }
}

#[test]
fn number_input_distinguishes_parser_zero_normalization_from_post_evaluation_rounding() {
    for expression in ["-0", "-0.0", "round(-0.25)", "0i - 0i"] {
        assert_eq!(
            original_real(expression).to_bits(),
            0_f64.to_bits(),
            "{expression}"
        );
        assert_eq!(
            evaluate_number_input(expression, NumberInputConstraints::default())
                .expect("source canonical zero")
                .to_bits(),
            0_f64.to_bits(),
            "{expression}"
        );
    }
    let rounded = evaluate_number_input(
        "-0.25",
        NumberInputConstraints {
            minimum: None,
            maximum: None,
            is_integer: true,
        },
    )
    .expect("widget rounds after source canonical evaluation");
    assert_eq!(rounded.to_bits(), (-0_f64).to_bits());
    assert_eq!(original_real("-0.25"), -0.25);
}

#[test]
fn number_input_failure_recovery_retains_caller_expression_and_constraints() {
    let constraints = NumberInputConstraints {
        minimum: Some(0.),
        maximum: Some(100.),
        is_integer: true,
    };
    let original_constraints = constraints;
    for malformed in ["50%", "2@", "0/0", "sqrt(-1)", "I", "100px"] {
        let expression = malformed.to_owned();
        let before = expression.clone();
        assert!(evaluate_number_input(&expression, constraints).is_err());
        assert_eq!(expression, before);
        assert_eq!(constraints, original_constraints);
        assert_eq!(evaluate_number_input("25 + 12.5", constraints), Ok(38.));
        assert_eq!(evaluate_number_input("25 + 12.5", constraints), Ok(38.));
    }
}

#[test]
fn mix_color_gamma_srgb_boundary_matches_original_channels_and_mode_opacity() {
    use graphite_core_types::{Color, blending::BlendMode, misc::Clampable as _};
    use graphite_raster_nodes::blending_nodes::blend_colors;

    let corpus = [
        ([0.2, 0.5, 0.9, 0.25], [0.8, 0.3, 0.1, 0.75]),
        ([0., 0., 0., 0.], [1., 1., 1., 1.]),
        ([1., 1., 1., 1.], [0., 0., 0., 0.]),
        ([0.7, 0.1, 0.3, -0.], [0.2, 0.9, 0.5, 1.]),
    ];
    for (over, under) in corpus {
        let [red, green, blue, alpha] = over;
        let original_over = Color::from_gamma_srgb_channels(red, green, blue, alpha);
        let [red, green, blue, alpha] = under;
        let original_under = Color::from_gamma_srgb_channels(red, green, blue, alpha);
        let original_input_bits = [over.map(f32::to_bits), under.map(f32::to_bits)];
        for mode in BlendMode::list().into_iter().flatten() {
            for opacity in [
                0.,
                25.,
                50.,
                100.,
                -10.,
                110.,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                let admitted = opacity.clamp_hard_min(0.).clamp_hard_max(100.);
                let expected = blend_colors(
                    original_over,
                    original_under,
                    *mode,
                    (admitted / 100.) as f32,
                )
                .to_gamma_srgb_channels();
                let result = crate::raster::evaluate_mix_color_gamma_srgb_channels(
                    over, under, *mode, opacity,
                )
                .expect("original generated Color Mix constructor");
                assert_eq!(
                    result.map(f32::to_bits),
                    expected.map(f32::to_bits),
                    "{mode:?}/{opacity}"
                );
                assert_eq!(
                    [over.map(f32::to_bits), under.map(f32::to_bits)],
                    original_input_bits
                );
            }
        }
    }
}
