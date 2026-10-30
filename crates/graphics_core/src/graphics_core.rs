pub mod document;
pub mod evaluate;
pub mod export;
pub mod formats;
pub mod numeric_input;
pub mod raster;
pub mod registry;
pub mod render;
pub mod resources;

use std::{
    fmt,
    ops::{Add, Mul},
    panic::AssertUnwindSafe,
};

use futures::FutureExt as _;
use graphite_core_types::{
    Context, Node,
    list::{Item, List},
    registry::DynFuture,
};
use graphite_glam::{DAffine2, DVec2};

struct Input<T>(T);

impl<'evaluation, T: Clone + Send + Sync + 'evaluation> Node<'evaluation, Context<'evaluation>>
    for Input<T>
{
    type Output = DynFuture<'evaluation, T>;

    fn eval(&'evaluation self, _context: Context<'evaluation>) -> Self::Output {
        Box::pin(async move { self.0.clone() })
    }
}

pub fn evaluate_add(augend: f64, addend: f64) -> f64 {
    evaluate_add_items(
        Item::new_from_element(augend),
        Item::new_from_element(addend),
    )
    .into_element()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddEvaluationError;

impl fmt::Display for AddEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Node graph execution panicked")
    }
}

impl std::error::Error for AddEvaluationError {}

pub fn evaluate_add_i64(augend: i64, addend: i64) -> Result<i64, AddEvaluationError> {
    evaluate_add_typed_items(
        Item::new_from_element(augend),
        Item::new_from_element(addend),
    )
    .map(Item::into_element)
}

pub fn evaluate_add_vector(
    augend: [f64; 2],
    addend: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    evaluate_add_typed_items(
        Item::new_from_element(DVec2::from_array(augend)),
        Item::new_from_element(DVec2::from_array(addend)),
    )
    .map(|result| result.into_element().to_array())
}

pub fn evaluate_add_scalar_vector(
    augend: f64,
    addend: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    evaluate_add_typed_items(
        Item::new_from_element(augend),
        Item::new_from_element(DVec2::from_array(addend)),
    )
    .map(|result| result.into_element().to_array())
}

pub fn evaluate_add_vector_scalar(
    augend: [f64; 2],
    addend: f64,
) -> Result<[f64; 2], AddEvaluationError> {
    evaluate_add_typed_items(
        Item::new_from_element(DVec2::from_array(augend)),
        Item::new_from_element(addend),
    )
    .map(|result| result.into_element().to_array())
}

fn evaluate_add_typed_items<A, B>(
    augend: Item<A>,
    addend: Item<B>,
) -> Result<Item<A::Output>, AddEvaluationError>
where
    A: Add<B> + Clone + Send + Sync + 'static,
    B: Clone + Send + Sync + 'static,
    A::Output: Send + Sync + 'static,
{
    let node = graphite_math_nodes::AddNode::new(Input(augend), Input(addend));
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

pub fn evaluate_add_i64_lists(
    augend: &[i64],
    addend: &[i64],
) -> Result<Vec<i64>, AddEvaluationError> {
    evaluate_add_typed_list_items(
        augend.iter().copied().map(Item::new_from_element).collect(),
        addend.iter().copied().map(Item::new_from_element).collect(),
    )
    .map(|result| result.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_add_vector_lists(
    augend: &[[f64; 2]],
    addend: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    evaluate_add_typed_list_items(vector_items(augend), vector_items(addend)).map(vector_elements)
}

pub fn evaluate_add_scalar_vector_lists(
    augend: &[f64],
    addend: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    evaluate_add_typed_list_items(
        augend.iter().copied().map(Item::new_from_element).collect(),
        vector_items(addend),
    )
    .map(vector_elements)
}

pub fn evaluate_add_vector_scalar_lists(
    augend: &[[f64; 2]],
    addend: &[f64],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    evaluate_add_typed_list_items(
        vector_items(augend),
        addend.iter().copied().map(Item::new_from_element).collect(),
    )
    .map(vector_elements)
}

fn vector_items(values: &[[f64; 2]]) -> List<DVec2> {
    values
        .iter()
        .copied()
        .map(DVec2::from_array)
        .map(Item::new_from_element)
        .collect()
}

fn vector_elements(values: List<DVec2>) -> Vec<[f64; 2]> {
    values
        .into_iter()
        .map(|item| item.into_element().to_array())
        .collect()
}

fn evaluate_add_typed_list_items<A, B>(
    augend: List<A>,
    addend: List<B>,
) -> Result<List<A::Output>, AddEvaluationError>
where
    A: Add<B> + Clone + Send + Sync + 'static,
    B: Clone + Send + Sync + 'static,
    A::Output: Clone + Send + Sync + 'static,
{
    let node = graphite_math_nodes::AddNodeMapped::new(Input(augend), Input(addend));
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

pub fn evaluate_add_lists(augend: &[f64], addend: &[f64]) -> Vec<f64> {
    let augend = augend.iter().copied().map(Item::new_from_element).collect();
    let addend = addend.iter().copied().map(Item::new_from_element).collect();
    evaluate_add_list_items(augend, addend)
        .into_iter()
        .map(Item::into_element)
        .collect()
}

fn evaluate_add_list_items(augend: List<f64>, addend: List<f64>) -> List<f64> {
    let node = graphite_math_nodes::AddNodeMapped::new(Input(augend), Input(addend));
    futures::executor::block_on(node.eval(Context::default()))
}

fn evaluate_add_items(augend: Item<f64>, addend: Item<f64>) -> Item<f64> {
    let node = graphite_math_nodes::AddNode::new(Input(augend), Input(addend));
    futures::executor::block_on(node.eval(Context::default()))
}

pub fn evaluate_sort_numbers(
    values: &[f64],
    sort_order: &[f64],
    reverse: bool,
) -> Result<Vec<f64>, AddEvaluationError> {
    evaluate_sort_number_items(
        values.iter().copied().map(Item::new_from_element).collect(),
        sort_order
            .iter()
            .copied()
            .map(Item::new_from_element)
            .collect(),
        reverse,
    )
    .map(|result| result.into_iter().map(Item::into_element).collect())
}

fn evaluate_sort_number_items(
    values: List<f64>,
    sort_order: List<f64>,
    reverse: bool,
) -> Result<List<f64>, AddEvaluationError> {
    let node = graphite_graphic_nodes::SortNode::new(
        Input(values),
        Input(sort_order),
        Input(Item::new_from_element(reverse)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

pub fn evaluate_multiply(multiplier: f64, multiplicand: f64) -> Result<f64, AddEvaluationError> {
    evaluate_multiply_items(
        Item::new_from_element(multiplier),
        Item::new_from_element(multiplicand),
    )
    .map(Item::into_element)
}

pub fn evaluate_multiply_lists(
    multiplier: &[f64],
    multiplicand: &[f64],
) -> Result<Vec<f64>, AddEvaluationError> {
    evaluate_multiply_list_items(
        multiplier
            .iter()
            .copied()
            .map(Item::new_from_element)
            .collect(),
        multiplicand
            .iter()
            .copied()
            .map(Item::new_from_element)
            .collect(),
    )
    .map(|result| result.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_multiply_i64(
    multiplier: i64,
    multiplicand: i64,
) -> Result<i64, AddEvaluationError> {
    evaluate_multiply_items(
        Item::new_from_element(multiplier),
        Item::new_from_element(multiplicand),
    )
    .map(Item::into_element)
}

pub fn evaluate_multiply_i64_lists(
    multiplier: &[i64],
    multiplicand: &[i64],
) -> Result<Vec<i64>, AddEvaluationError> {
    evaluate_multiply_list_items(
        multiplier
            .iter()
            .copied()
            .map(Item::new_from_element)
            .collect(),
        multiplicand
            .iter()
            .copied()
            .map(Item::new_from_element)
            .collect(),
    )
    .map(|result| result.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_multiply_vector(
    multiplier: [f64; 2],
    multiplicand: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    evaluate_multiply_items(
        Item::new_from_element(DVec2::from_array(multiplier)),
        Item::new_from_element(DVec2::from_array(multiplicand)),
    )
    .map(|item| item.into_element().to_array())
}

pub fn evaluate_multiply_vector_lists(
    multiplier: &[[f64; 2]],
    multiplicand: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    evaluate_multiply_list_items(vector_items(multiplier), vector_items(multiplicand)).map(
        |result| {
            result
                .into_iter()
                .map(|item| item.into_element().to_array())
                .collect()
        },
    )
}

pub fn evaluate_multiply_scalar_vector(
    multiplier: f64,
    multiplicand: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    evaluate_multiply_items(
        Item::new_from_element(multiplier),
        Item::new_from_element(DVec2::from_array(multiplicand)),
    )
    .map(|item| item.into_element().to_array())
}

pub fn evaluate_multiply_scalar_vector_lists(
    multiplier: &[f64],
    multiplicand: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    evaluate_multiply_list_items(
        multiplier
            .iter()
            .copied()
            .map(Item::new_from_element)
            .collect(),
        vector_items(multiplicand),
    )
    .map(|result| {
        result
            .into_iter()
            .map(|item| item.into_element().to_array())
            .collect()
    })
}

pub fn evaluate_multiply_vector_scalar(
    multiplier: [f64; 2],
    multiplicand: f64,
) -> Result<[f64; 2], AddEvaluationError> {
    evaluate_multiply_items(
        Item::new_from_element(DVec2::from_array(multiplier)),
        Item::new_from_element(multiplicand),
    )
    .map(|item| item.into_element().to_array())
}

pub fn evaluate_multiply_vector_scalar_lists(
    multiplier: &[[f64; 2]],
    multiplicand: &[f64],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    evaluate_multiply_list_items(
        vector_items(multiplier),
        multiplicand
            .iter()
            .copied()
            .map(Item::new_from_element)
            .collect(),
    )
    .map(|result| {
        result
            .into_iter()
            .map(|item| item.into_element().to_array())
            .collect()
    })
}

pub fn evaluate_multiply_affine(
    multiplier: [f64; 6],
    multiplicand: [f64; 6],
) -> Result<[f64; 6], AddEvaluationError> {
    evaluate_multiply_items(
        Item::new_from_element(DAffine2::from_cols_array(&multiplier)),
        Item::new_from_element(DAffine2::from_cols_array(&multiplicand)),
    )
    .map(|item| item.into_element().to_cols_array())
}

pub fn evaluate_multiply_affine_lists(
    multiplier: &[[f64; 6]],
    multiplicand: &[[f64; 6]],
) -> Result<Vec<[f64; 6]>, AddEvaluationError> {
    evaluate_multiply_list_items(
        multiplier
            .iter()
            .map(|value| Item::new_from_element(DAffine2::from_cols_array(value)))
            .collect(),
        multiplicand
            .iter()
            .map(|value| Item::new_from_element(DAffine2::from_cols_array(value)))
            .collect(),
    )
    .map(|result| {
        result
            .into_iter()
            .map(|item| item.into_element().to_cols_array())
            .collect()
    })
}

fn evaluate_multiply_items<A, B>(
    multiplier: Item<A>,
    multiplicand: Item<B>,
) -> Result<Item<A::Output>, AddEvaluationError>
where
    A: Mul<B> + Clone + Send + Sync + 'static,
    B: Clone + Send + Sync + 'static,
    A::Output: Send + Sync + 'static,
{
    let node = graphite_math_nodes::MultiplyNode::new(Input(multiplier), Input(multiplicand));
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}
fn evaluate_multiply_list_items<A, B>(
    multiplier: List<A>,
    multiplicand: List<B>,
) -> Result<List<A::Output>, AddEvaluationError>
where
    A: Mul<B> + Clone + Send + Sync + 'static,
    B: Clone + Send + Sync + 'static,
    A::Output: Clone + Send + Sync + 'static,
{
    let node = graphite_math_nodes::MultiplyNodeMapped::new(Input(multiplier), Input(multiplicand));
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

#[cfg(test)]
mod tests {
    use super::{
        Item, List, evaluate_add, evaluate_add_items, evaluate_add_list_items, evaluate_add_lists,
    };

    #[test]
    fn at_gra_node_084_2_scalar_ieee_values_and_signed_zero() {
        assert_eq!(super::evaluate_multiply(-2.5, 4.0), Ok(-10.0));
        assert_eq!(super::evaluate_multiply(0.0, 4.0), Ok(0.0));
        let negative_zero =
            super::evaluate_multiply(-0.0, 2.0).expect("original product evaluates");
        assert_eq!(negative_zero.to_bits(), (-0.0_f64).to_bits());
        assert!(
            super::evaluate_multiply(f64::INFINITY, 0.0)
                .expect("IEEE product evaluates")
                .is_nan()
        );
        assert!(
            super::evaluate_multiply(f64::NAN, 2.0)
                .expect("IEEE product evaluates")
                .is_nan()
        );
        assert_eq!(super::evaluate_multiply(f64::MAX, 2.0), Ok(f64::INFINITY));
        let result = super::evaluate_multiply_lists(&[-0.0, f64::INFINITY], &[2.0, 0.0])
            .expect("original mapped product evaluates");
        assert_eq!(
            result.first().map(|value| value.to_bits()),
            Some((-0.0_f64).to_bits())
        );
        assert!(result.get(1).is_some_and(|value| value.is_nan()));
    }

    #[test]
    fn at_gra_node_084_2_integer_precision_and_source_overflow_are_profile_specific() {
        assert_eq!(
            super::evaluate_multiply_i64(9_007_199_254_740_993, 3),
            Ok(27_021_597_764_222_979)
        );
        assert_eq!(super::evaluate_multiply_i64(i64::MAX, 1), Ok(i64::MAX));
        assert_eq!(super::evaluate_multiply_i64(i64::MIN, 1), Ok(i64::MIN));
        assert_eq!(super::evaluate_multiply_i64(-5, -3), Ok(15));
        let positive = super::evaluate_multiply_i64(i64::MAX, 2);
        let negative = super::evaluate_multiply_i64(i64::MIN, -1);
        let multiplier = [2, i64::MAX];
        let mapped = super::evaluate_multiply_i64_lists(&multiplier, &[3, 2]);
        if cfg!(debug_assertions) {
            assert_eq!(positive, Err(super::AddEvaluationError));
            assert_eq!(negative, Err(super::AddEvaluationError));
            assert_eq!(mapped, Err(super::AddEvaluationError));
        } else {
            assert_eq!(positive, Ok(-2));
            assert_eq!(negative, Ok(i64::MIN));
            assert_eq!(mapped, Ok(vec![6, -2]));
        }
        assert_eq!(multiplier, [2, i64::MAX]);
        assert_eq!(
            super::evaluate_multiply_i64_lists(&[2, 3], &[4]),
            Ok(vec![8, 12])
        );
        assert_eq!(
            super::evaluate_multiply_i64_lists(&[9_007_199_254_740_993], &[3, 2]),
            Ok(vec![27_021_597_764_222_979, 18_014_398_509_481_986])
        );
    }

    #[test]
    fn at_gra_node_084_2_vector_pairs_and_mapped_order_use_original_nodes() {
        assert_eq!(
            super::evaluate_multiply_vector([2.0, -3.0], [4.0, 5.0]),
            Ok([8.0, -15.0])
        );
        assert_eq!(
            super::evaluate_multiply_scalar_vector(-2.0, [4.0, 5.0]),
            Ok([-8.0, -10.0])
        );
        assert_eq!(
            super::evaluate_multiply_vector_scalar([4.0, 5.0], -2.0),
            Ok([-8.0, -10.0])
        );
        assert_eq!(
            super::evaluate_multiply_vector_lists(&[[2.0, -3.0]], &[[4.0, 5.0], [-2.0, 3.0]]),
            Ok(vec![[8.0, -15.0], [-4.0, -9.0]])
        );
        assert_eq!(
            super::evaluate_multiply_vector_lists(&[[4.0, 5.0], [-2.0, 3.0]], &[[2.0, -3.0]]),
            Ok(vec![[8.0, -15.0], [-4.0, -9.0]])
        );
        assert_eq!(
            super::evaluate_multiply_scalar_vector_lists(&[2.0], &[[4.0, 5.0], [-2.0, 3.0]]),
            Ok(vec![[8.0, 10.0], [-4.0, 6.0]])
        );
        assert_eq!(
            super::evaluate_multiply_scalar_vector_lists(&[2.0, -3.0], &[[4.0, 5.0]]),
            Ok(vec![[8.0, 10.0], [-12.0, -15.0]])
        );
        assert_eq!(
            super::evaluate_multiply_vector_scalar_lists(&[[4.0, 5.0]], &[2.0, -3.0]),
            Ok(vec![[8.0, 10.0], [-12.0, -15.0]])
        );
        assert_eq!(
            super::evaluate_multiply_vector_scalar_lists(&[[4.0, 5.0], [-2.0, 3.0]], &[2.0]),
            Ok(vec![[8.0, 10.0], [-4.0, 6.0]])
        );
        assert_eq!(
            super::evaluate_multiply_lists(&[2.0, -3.0], &[4.0]),
            Ok(vec![8.0, -12.0])
        );
        assert_eq!(
            super::evaluate_multiply_lists(&[4.0], &[2.0, -3.0]),
            Ok(vec![8.0, -12.0])
        );
    }

    #[test]
    fn at_gra_node_084_2_affine_composition_preserves_source_operand_and_column_order() {
        let first = [2.0, 0.0, 0.0, 3.0, 5.0, 7.0];
        let second = [0.0, 1.0, -1.0, 0.0, 11.0, 13.0];
        let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let forward = [0.0, 3.0, -2.0, 0.0, 27.0, 46.0];
        let backward = [0.0, 2.0, -3.0, 0.0, 4.0, 18.0];
        assert_eq!(super::evaluate_multiply_affine(first, second), Ok(forward));
        assert_eq!(super::evaluate_multiply_affine(second, first), Ok(backward));
        assert_eq!(
            super::evaluate_multiply_affine_lists(&[first], &[second, identity]),
            Ok(vec![forward, first])
        );
        assert_eq!(
            super::evaluate_multiply_affine_lists(&[first, identity], &[second]),
            Ok(vec![forward, second])
        );
        assert_eq!(
            super::evaluate_multiply_affine([0.0; 6], second),
            Ok([0.0; 6])
        );
    }

    #[test]
    fn at_gra_node_084_2_all_six_list_pairs_obey_source_empty_rules() {
        for (left, right) in [
            (&[][..], &[1.0][..]),
            (&[1.0][..], &[][..]),
            (&[][..], &[][..]),
        ] {
            assert_eq!(super::evaluate_multiply_lists(left, right), Ok(vec![]));
        }
        for (left, right) in [(&[][..], &[1][..]), (&[1][..], &[][..]), (&[][..], &[][..])] {
            assert_eq!(super::evaluate_multiply_i64_lists(left, right), Ok(vec![]));
        }
        for (left, right) in [
            (&[][..], &[[1.0, 2.0]][..]),
            (&[[1.0, 2.0]][..], &[][..]),
            (&[][..], &[][..]),
        ] {
            assert_eq!(
                super::evaluate_multiply_vector_lists(left, right),
                Ok(vec![])
            );
        }
        assert_eq!(
            super::evaluate_multiply_scalar_vector_lists(&[], &[[1.0, 2.0]]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_multiply_scalar_vector_lists(&[1.0], &[]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_multiply_scalar_vector_lists(&[], &[]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_multiply_vector_scalar_lists(&[], &[1.0]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_multiply_vector_scalar_lists(&[[1.0, 2.0]], &[]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_multiply_vector_scalar_lists(&[], &[]),
            Ok(vec![])
        );
        let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        for (left, right) in [
            (&[][..], &[identity][..]),
            (&[identity][..], &[][..]),
            (&[][..], &[][..]),
        ] {
            assert_eq!(
                super::evaluate_multiply_affine_lists(left, right),
                Ok(vec![])
            );
        }
    }

    #[test]
    fn at_gra_node_084_2_multiply_preserves_left_item_and_repeated_slot_attributes() {
        let first = super::DAffine2::from_cols_array(&[2.0, 0.0, 0.0, 3.0, 5.0, 7.0]);
        let second = super::DAffine2::from_cols_array(&[0.0, 1.0, -1.0, 0.0, 11.0, 13.0]);
        let left = Item::new_from_element(first).with_attribute("owner", String::from("left"));
        let right = Item::new_from_element(second).with_attribute("right-only", true);
        let result = super::evaluate_multiply_items(left.clone(), right.clone())
            .expect("original affine item product evaluates");
        assert_eq!(
            result.attribute::<String>("owner").map(String::as_str),
            Some("left")
        );
        assert!(result.attribute::<bool>("right-only").is_none());
        let left_list: List<_> = [left.clone()].into_iter().collect();
        let right_list: List<_> = [right, Item::new_from_element(super::DAffine2::IDENTITY)]
            .into_iter()
            .collect();
        let result = super::evaluate_multiply_list_items(left_list.clone(), right_list.clone())
            .expect("original affine mapped product evaluates");
        assert_eq!(result.len(), 2);
        for index in 0..2 {
            assert_eq!(
                result
                    .attribute::<String>("owner", index)
                    .map(String::as_str),
                Some("left")
            );
            assert!(result.attribute::<bool>("right-only", index).is_none());
        }
        assert_eq!(
            left_list
                .iter_element_values()
                .map(|value| value.to_cols_array())
                .collect::<Vec<_>>(),
            vec![first.to_cols_array()]
        );
        assert_eq!(
            right_list
                .iter_element_values()
                .map(|value| value.to_cols_array())
                .collect::<Vec<_>>(),
            vec![
                second.to_cols_array(),
                super::DAffine2::IDENTITY.to_cols_array()
            ]
        );
    }

    #[test]
    fn at_gra_node_063_2_numeric_sort_has_source_empty_and_natural_order() {
        assert_eq!(super::evaluate_sort_numbers(&[], &[], false), Ok(vec![]));
        assert_eq!(super::evaluate_sort_numbers(&[], &[1.0], true), Ok(vec![]));
        assert_eq!(
            super::evaluate_sort_numbers(&[2.0], &[], true),
            Ok(vec![2.0])
        );
        let values = [3.0, -2.0, 3.0, 0.0];
        assert_eq!(
            super::evaluate_sort_numbers(&values, &[], false),
            Ok(vec![-2.0, 0.0, 3.0, 3.0])
        );
        assert_eq!(
            super::evaluate_sort_numbers(&values, &[], true),
            Ok(vec![3.0, 3.0, 0.0, -2.0])
        );
        assert_eq!(values, [3.0, -2.0, 3.0, 0.0]);
    }

    #[test]
    fn at_gra_node_063_2_numeric_keys_repeat_last_ignore_extra_and_keep_ties() {
        let values = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(
            super::evaluate_sort_numbers(&values, &[2.0, 1.0], false),
            Ok(vec![20.0, 30.0, 40.0, 10.0])
        );
        assert_eq!(
            super::evaluate_sort_numbers(&values, &[2.0, 1.0], true),
            Ok(values.to_vec())
        );
        assert_eq!(
            super::evaluate_sort_numbers(&values, &[1.0], true),
            Ok(values.to_vec())
        );
        assert_eq!(
            super::evaluate_sort_numbers(&values, &[3.0, 2.0, 1.0, 0.0, -10.0], false),
            Ok(vec![40.0, 30.0, 20.0, 10.0])
        );
    }

    #[test]
    fn at_gra_node_063_2_sort_uses_total_float_order_for_elements_and_keys() {
        let first_nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let second_nan = f64::from_bits(0x7ff8_0000_0000_0002);
        let values = [
            second_nan,
            0.0,
            f64::INFINITY,
            -0.0,
            first_nan,
            f64::NEG_INFINITY,
        ];
        let expected = [
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            f64::INFINITY,
            first_nan,
            second_nan,
        ];
        for reverse in [false, true] {
            let result = super::evaluate_sort_numbers(&values, &[], reverse)
                .expect("original numeric Sort evaluates");
            let mut expected_bits = expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            if reverse {
                expected_bits.reverse();
            }
            assert_eq!(
                result
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_bits
            );
        }
        assert_eq!(
            super::evaluate_sort_numbers(
                &[10.0, 20.0, 30.0, 40.0],
                &[first_nan, 0.0, -0.0, f64::INFINITY],
                false
            ),
            Ok(vec![30.0, 20.0, 40.0, 10.0])
        );
    }

    #[test]
    fn at_gra_node_063_2_sort_moves_complete_items_without_key_attributes() {
        let values: List<f64> = [2.0, 1.0, 2.0]
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                Item::new_from_element(value).with_attribute("original-index", index)
            })
            .collect();
        let keys: List<f64> = [1.0, 1.0, 0.0]
            .into_iter()
            .map(|value| Item::new_from_element(value).with_attribute("key-only", true))
            .collect();
        for (reverse, expected) in [(false, vec![2, 0, 1]), (true, vec![0, 1, 2])] {
            let result = super::evaluate_sort_number_items(values.clone(), keys.clone(), reverse)
                .expect("original Sort retains whole items");
            assert_eq!(result.len(), values.len());
            for (position, original_index) in expected.into_iter().enumerate() {
                assert_eq!(
                    result.attribute::<usize>("original-index", position),
                    Some(&original_index)
                );
                assert!(result.attribute::<bool>("key-only", position).is_none());
            }
        }
        assert_eq!(
            values.iter_element_values().copied().collect::<Vec<_>>(),
            vec![2.0, 1.0, 2.0]
        );
        assert_eq!(
            keys.iter_element_values().copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 0.0]
        );
    }

    #[test]
    fn at_gra_node_077_2_typed_lists_broadcast_and_preserve_order() {
        assert_eq!(
            super::evaluate_add_i64_lists(&[9_007_199_254_740_993, -5, 0], &[2, 3]),
            Ok(vec![9_007_199_254_740_995, -2, 3])
        );
        assert_eq!(
            super::evaluate_add_i64_lists(&[2], &[3, -5]),
            Ok(vec![5, -3])
        );
        assert_eq!(
            super::evaluate_add_vector_lists(&[[1.0, -2.0]], &[[3.0, 5.0], [-3.0, 7.0]]),
            Ok(vec![[4.0, 3.0], [-2.0, 5.0]])
        );
        assert_eq!(
            super::evaluate_add_vector_lists(&[[1.0, -2.0], [-3.0, 7.0]], &[[3.0, 5.0]]),
            Ok(vec![[4.0, 3.0], [0.0, 12.0]])
        );
        assert_eq!(
            super::evaluate_add_scalar_vector_lists(&[2.0], &[[3.0, -5.0], [-3.0, 7.0]]),
            Ok(vec![[5.0, -3.0], [-1.0, 9.0]])
        );
        assert_eq!(
            super::evaluate_add_scalar_vector_lists(&[2.0, -3.0], &[[3.0, -5.0]]),
            Ok(vec![[5.0, -3.0], [0.0, -8.0]])
        );
        assert_eq!(
            super::evaluate_add_vector_scalar_lists(&[[3.0, -5.0]], &[2.0, -3.0]),
            Ok(vec![[5.0, -3.0], [0.0, -8.0]])
        );
        assert_eq!(
            super::evaluate_add_vector_scalar_lists(&[[3.0, -5.0], [-3.0, 7.0]], &[2.0]),
            Ok(vec![[5.0, -3.0], [-1.0, 9.0]])
        );
    }

    #[test]
    fn at_gra_node_077_2_every_typed_list_pair_has_source_empty_behavior() {
        for (augend, addend) in [(&[][..], &[1][..]), (&[1][..], &[][..]), (&[][..], &[][..])] {
            assert_eq!(super::evaluate_add_i64_lists(augend, addend), Ok(vec![]));
        }
        assert_eq!(
            super::evaluate_add_vector_lists(&[], &[[1.0, 2.0]]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_add_vector_lists(&[[1.0, 2.0]], &[]),
            Ok(vec![])
        );
        assert_eq!(super::evaluate_add_vector_lists(&[], &[]), Ok(vec![]));
        assert_eq!(
            super::evaluate_add_scalar_vector_lists(&[], &[[1.0, 2.0]]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_add_scalar_vector_lists(&[1.0], &[]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_add_scalar_vector_lists(&[], &[]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_add_vector_scalar_lists(&[], &[1.0]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_add_vector_scalar_lists(&[[1.0, 2.0]], &[]),
            Ok(vec![])
        );
        assert_eq!(
            super::evaluate_add_vector_scalar_lists(&[], &[]),
            Ok(vec![])
        );
    }

    #[test]
    fn at_gra_node_077_2_integer_list_late_overflow_is_atomic() {
        let result = super::evaluate_add_i64_lists(&[5, i64::MAX], &[2, 1]);
        if cfg!(debug_assertions) {
            assert_eq!(result, Err(super::AddEvaluationError));
        } else {
            assert_eq!(result, Ok(vec![7, i64::MIN]));
        }
        assert_eq!(
            super::evaluate_add_i64_lists(&[5, -2], &[2, 1]),
            Ok(vec![7, -1])
        );
    }

    #[test]
    fn at_gra_node_077_2_typed_list_slots_preserve_left_attributes_and_inputs() {
        let scalar: List<i64> =
            [Item::new_from_element(2_i64).with_attribute("name", String::from("left"))]
                .into_iter()
                .collect();
        let addend: List<i64> = [3, -5]
            .into_iter()
            .map(|value| Item::new_from_element(value).with_attribute("right-only", true))
            .collect();
        let result = super::evaluate_add_typed_list_items(scalar.clone(), addend.clone());
        let Ok(result) = result else {
            panic!("registered integer List Add failed")
        };
        assert_eq!(
            result.iter_element_values().copied().collect::<Vec<_>>(),
            vec![5, -3]
        );
        for index in 0..2 {
            assert_eq!(
                result
                    .attribute::<String>("name", index)
                    .map(String::as_str),
                Some("left")
            );
            assert!(result.attribute::<bool>("right-only", index).is_none());
        }
        assert_eq!(
            scalar.iter_element_values().copied().collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(
            addend.iter_element_values().copied().collect::<Vec<_>>(),
            vec![3, -5]
        );
        let vector: List<super::DVec2> = [Item::new_from_element(super::DVec2::new(3.0, -5.0))
            .with_attribute("name", String::from("vector"))]
        .into_iter()
        .collect();
        let scalar: List<f64> = [2.0, -3.0]
            .into_iter()
            .map(|value| Item::new_from_element(value).with_attribute("right-only", true))
            .collect();
        let result = super::evaluate_add_typed_list_items(vector.clone(), scalar.clone());
        let Ok(result) = result else {
            panic!("registered mixed List Add failed")
        };
        assert_eq!(
            result
                .iter_element_values()
                .map(|value| value.to_array())
                .collect::<Vec<_>>(),
            vec![[5.0, -3.0], [0.0, -8.0]]
        );
        for index in 0..2 {
            assert_eq!(
                result
                    .attribute::<String>("name", index)
                    .map(String::as_str),
                Some("vector")
            );
            assert!(result.attribute::<bool>("right-only", index).is_none());
        }
        assert_eq!(
            vector
                .iter_element_values()
                .map(|value| value.to_array())
                .collect::<Vec<_>>(),
            vec![[3.0, -5.0]]
        );
        assert_eq!(
            scalar.iter_element_values().copied().collect::<Vec<_>>(),
            vec![2.0, -3.0]
        );
    }

    #[test]
    fn at_gra_node_077_2_integer_precision_and_boundaries() {
        assert_eq!(
            super::evaluate_add_i64(9_007_199_254_740_993, 2),
            Ok(9_007_199_254_740_995)
        );
        assert_eq!(super::evaluate_add_i64(i64::MAX, 0), Ok(i64::MAX));
        assert_eq!(super::evaluate_add_i64(i64::MIN, 0), Ok(i64::MIN));
        assert_eq!(super::evaluate_add_i64(-5, 3), Ok(-2));
    }

    #[test]
    fn at_gra_node_077_2_integer_overflow_preserves_profile_behavior() {
        let positive = super::evaluate_add_i64(i64::MAX, 1);
        let negative = super::evaluate_add_i64(i64::MIN, -1);
        if cfg!(debug_assertions) {
            assert_eq!(positive, Err(super::AddEvaluationError));
            assert_eq!(negative, Err(super::AddEvaluationError));
            assert_eq!(
                super::AddEvaluationError.to_string(),
                "Node graph execution panicked"
            );
        } else {
            assert_eq!(positive, Ok(i64::MIN));
            assert_eq!(negative, Ok(i64::MAX));
        }
        assert_eq!(super::evaluate_add_i64(5, -2), Ok(3));
    }

    #[test]
    fn at_gra_node_077_2_vectors_and_both_mixed_directions() {
        assert_eq!(
            super::evaluate_add_vector([1.0, -2.0], [3.0, 5.0]),
            Ok([4.0, 3.0])
        );
        assert_eq!(
            super::evaluate_add_scalar_vector(2.0, [3.0, -5.0]),
            Ok([5.0, -3.0])
        );
        assert_eq!(
            super::evaluate_add_vector_scalar([3.0, -5.0], 2.0),
            Ok([5.0, -3.0])
        );
        let result = super::evaluate_add_vector([-0.0, f64::INFINITY], [-0.0, f64::NEG_INFINITY]);
        let Ok([first, second]) = result else {
            panic!("registered vector Add failed")
        };
        assert_eq!(first.to_bits(), (-0.0_f64).to_bits());
        assert!(second.is_nan());
    }

    #[test]
    fn at_gra_node_077_2_typed_add_preserves_left_attributes() {
        let augend = Item::new_from_element(2_i64).with_attribute("name", String::from("left"));
        let addend = Item::new_from_element(3_i64).with_attribute("right-only", true);
        let result = super::evaluate_add_typed_items(augend.clone(), addend.clone());
        let Ok(result) = result else {
            panic!("registered integer Add failed")
        };
        assert_eq!(*result.element(), 5);
        assert_eq!(
            result.attribute::<String>("name").map(String::as_str),
            Some("left")
        );
        assert!(result.attribute::<bool>("right-only").is_none());
        assert_eq!(*augend.element(), 2);
        assert_eq!(*addend.element(), 3);
        let scalar = Item::new_from_element(2.0_f64).with_attribute("name", String::from("scalar"));
        let vector =
            Item::new_from_element(super::DVec2::new(3.0, -5.0)).with_attribute("right-only", true);
        let result = super::evaluate_add_typed_items(scalar.clone(), vector.clone());
        let Ok(result) = result else {
            panic!("registered mixed Add failed")
        };
        assert_eq!(result.element().to_array(), [5.0, -3.0]);
        assert_eq!(
            result.attribute::<String>("name").map(String::as_str),
            Some("scalar")
        );
        assert!(result.attribute::<bool>("right-only").is_none());
        assert_eq!(*scalar.element(), 2.0);
        assert_eq!(vector.element().to_array(), [3.0, -5.0]);
    }

    #[test]
    fn at_gra_node_077_2_lists_use_source_broadcast_and_empty_rules() {
        assert_eq!(
            evaluate_add_lists(&[1.0, -2.0, 0.0], &[10.0, 20.0]),
            vec![11.0, 18.0, 20.0]
        );
        assert_eq!(
            evaluate_add_lists(&[1.0], &[10.0, 20.0, -2.0]),
            vec![11.0, 21.0, -1.0]
        );
        assert!(evaluate_add_lists(&[], &[1.0]).is_empty());
        assert!(evaluate_add_lists(&[1.0], &[]).is_empty());
        assert!(evaluate_add_lists(&[], &[]).is_empty());
    }

    #[test]
    fn at_gra_node_077_2_lists_preserve_slot_attributes() {
        let augend: List<f64> = [
            Item::new_from_element(1.0).with_attribute("name", String::from("first")),
            Item::new_from_element(-2.0).with_attribute("name", String::from("last")),
        ]
        .into_iter()
        .collect();
        let addend: List<f64> = [10.0, 20.0, 30.0]
            .into_iter()
            .map(|value| Item::new_from_element(value).with_attribute("right-only", true))
            .collect();
        let result = evaluate_add_list_items(augend.clone(), addend.clone());
        assert_eq!(
            result.iter_element_values().copied().collect::<Vec<_>>(),
            vec![11.0, 18.0, 28.0]
        );
        for (index, name) in ["first", "last", "last"].into_iter().enumerate() {
            assert_eq!(
                result
                    .attribute::<String>("name", index)
                    .map(String::as_str),
                Some(name)
            );
            assert!(result.attribute::<bool>("right-only", index).is_none());
        }
        assert_eq!(
            augend.iter_element_values().copied().collect::<Vec<_>>(),
            vec![1.0, -2.0]
        );
        assert_eq!(
            addend.iter_element_values().copied().collect::<Vec<_>>(),
            vec![10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn at_gra_node_077_2_add_preserves_augend_attributes() {
        let augend = Item::new_from_element(-5.0)
            .with_attribute("name", String::from("left"))
            .with_attribute("opacity", 0.25_f64);
        let addend = Item::new_from_element(3.0)
            .with_attribute("name", String::from("right"))
            .with_attribute("right-only", 17_i64);
        let result = evaluate_add_items(augend.clone(), addend.clone());
        assert_eq!(*result.element(), -2.0);
        assert_eq!(
            result.attribute::<String>("name"),
            Some(&String::from("left"))
        );
        assert_eq!(result.attribute::<f64>("opacity"), Some(&0.25));
        assert!(result.attribute::<i64>("right-only").is_none());
        assert_eq!(*augend.element(), -5.0);
        assert_eq!(*addend.element(), 3.0);
        assert_eq!(addend.attribute::<i64>("right-only"), Some(&17));
    }

    #[test]
    fn at_r02_1_native_source_add_handles_scalar_boundaries() {
        assert_eq!(evaluate_add(2.0, 3.0), 5.0);
        assert_eq!(evaluate_add(-5.0, 3.0), -2.0);
        assert_eq!(evaluate_add(0.0, 0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(evaluate_add(-0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
        assert!(evaluate_add(f64::INFINITY, f64::NEG_INFINITY).is_nan());
        assert_eq!(evaluate_add(f64::MAX, f64::MAX), f64::INFINITY);
    }
}

#[cfg(test)]
mod conformance;
