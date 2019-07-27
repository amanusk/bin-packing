#![feature(test)]

use itertools::Itertools;
use std::sync::{Arc, RwLock};

#[derive(Clone, PartialOrd, Debug)]
pub struct Bin {
    items: Vec<f32>,
    capacity: f32,
    total: f32,
    residue: f32,
}

impl PartialEq for Bin {
    fn eq(&self, other: &Bin) -> bool {
        self.items == other.items
            && self.capacity == other.capacity
            && self.total == other.total
            && self.residue == other.residue
    }
}

impl Bin {
    pub fn new(capacity: f32) -> Bin {
        Bin {
            items: Vec::new(),
            capacity: capacity,
            total: 0.0,
            residue: capacity,
        }
    }

    pub fn add(&mut self, item: f32) {
        if item + self.total > self.capacity {
            panic!("Cannot add item, capacity overflow");
        }
        self.items.push(item);
        self.total += item;
        self.residue -= item;
        assert_eq!(self.total + self.residue, self.capacity);
    }

    pub fn can_fit(&self, item: f32) -> bool {
        item + self.total <= self.capacity
    }

    pub fn capacity(&self) -> f32 {
        return self.capacity;
    }

    pub fn total(&self) -> f32 {
        return self.total;
    }

    pub fn residue(&self) -> f32 {
        return self.residue;
    }

    pub fn items(&self) -> &Vec<f32> {
        return &self.items;
    }
}

// Computes the absolute lower bound of bins possible
fn simple_lower_bound(items: &Vec<f32>, bin_capacity: f32) -> f32 {
    let sum_items: f32 = items.iter().sum();
    sum_items / bin_capacity
}

// Computes a tighter limit on the lower bound of minimum bins possbile
fn l2_lower_bound(items: &Vec<f32>, bin_capacity: f32) -> f32 {
    let mut sorted_items = items.clone();
    sorted_items.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut overflow = 0.0;
    let mut estimated_waste = 0.0;
    while sorted_items.len() > 0 {
        let largest = sorted_items.remove(0);
        let residue = bin_capacity - largest;
        let mut sum_of_less_than = overflow;
        while let Some(last) = sorted_items.last().copied() {
            if last <= residue {
                let smallest = sorted_items.pop().unwrap();
                sum_of_less_than += smallest;
            } else {
                break;
            }
        }
        if sum_of_less_than == residue {
            continue;
        } else if sum_of_less_than < residue {
            estimated_waste += residue - sum_of_less_than;
            overflow = 0.0;
        } else {
            overflow = sum_of_less_than - residue;
        }
    }
    let sum_items = items.iter().fold(0.0, |sum, x| sum + x) + estimated_waste;
    sum_items / bin_capacity
}

fn eq_with_nan_eq(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || (a == b)
}

fn vec_compare(va: &[f32], vb: &[f32]) -> bool {
    (va.len() == vb.len()) &&  // zip stops at the shortest
     va.iter()
       .zip(vb)
       .all(|(a,b)| eq_with_nan_eq(*a,*b))
}

fn is_dominant(v1: &Vec<f32>, v2: &Vec<f32>) -> bool {
    if v1.len() != v2.len() {
        return false;
    }
    if v1.len() == 1 {
        println!("Comapring {}, {}", v1[0], v2[0]);
        return v1[0] > v2[0];
    }
    false
}

fn dominated(v1: &Vec<f32>, all: &Vec<&Vec<f32>>) -> bool {
    let dominated = false;
    for v in all.iter() {
        if is_dominant(v, v1) {
            return true;
        }
    }

    return dominated;
}

pub fn get_all_undominated(items: &Vec<f32>, limit: f32) {
    let mut all_possible = Vec::new();
    for i in 0..items.len() {
        all_possible.extend(
            items.iter().combinations(i).filter(|x: &Vec<&f32>| {
                x.iter().fold(0.0, |sum, y| sum + *y) <= limit && x.len() > 0
            }),
        );
    }
    let mut all_possible_values = Vec::new();
    for vec in all_possible.iter() {
        let mut new_vec = Vec::new();
        for item in vec {
            new_vec.push(**item);
        }
        all_possible_values.push(new_vec.clone());
    }
    // for (a, b) in all_possible.iter().tuple_combinations() {
    //     println!("{:?}, {:?}", a, b);
    // }
    // println!("All possible");
    // all_possible_values.iter().for_each(|x| println!("{:?}", x));
    // let mut undominated = Vec::new();
    // all_possible_values.iter().for_each(|x: &Vec<f32>| {
    //     if !dominated(x, all_possible) {
    //         println!("Adding {:?}", x);
    //         undominated.push(x);
    //     }
    // });
    // println!("Undomintated");
    // undominated.iter().for_each(|x| println!("{:?}", x));
}

pub fn stupid_bin_packing_outer(items: &Vec<f32>, bin_capacity: f32) -> Vec<Bin> {
    // Set upper bound to the number of items
    let best_current = Arc::new(RwLock::new(Vec::new()));
    //println!(
    //    "Simple lower bound {}",
    //    simple_lower_bound(&items, bin_capacity).ceil()
    //);
    let l2_lower_bound = l2_lower_bound(&items, bin_capacity).ceil();
    //println!("l2 lower bound {}", l2_lower_bound);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

    // Option to create worst solution
    // for item in items.clone() {
    //     let mut bin = Bin::new(bin_capacity);
    //     bin.add(item);
    //     best_current.read().unwrap().push(bin);
    // }

    *best_current.write().unwrap() = best_fit_first(&items, bin_capacity);
    //println!(
    //    "Best fit first: {} bins",
    //    &best_current.read().unwrap().len(),
    //);
    //println!("Starting from {:?}", best_current.read().unwrap());
    //println!("Best fit first {:?}", &best_current.read().unwrap());
    assert!(best_current.read().unwrap().len() >= l2_lower_bound as usize);
    if best_current.read().unwrap().len() <= l2_lower_bound as usize {
        // Finish
        // println!("Best found {:?}", &best_current);
        // println!("Best found bins {:?}", &best_current.read().unwrap().len());
        return Arc::clone(&best_current).read().unwrap().clone();
    }

    // Create a new item list for every recursion, so they could (potentially) happen in parallel
    let cloned_best = Arc::clone(&best_current);
    pool.install(move || {
        stupid_bin_packing(items.clone(), Vec::new(), bin_capacity, cloned_best);
    });
    // println!("Best found {:?}", &best_current);
    // println!("Best found bins {:?}", &best_current.read().unwrap().len());
    Arc::clone(&best_current).read().unwrap().clone()
}

fn stupid_bin_packing(
    items: Vec<f32>,
    filled_bins: Vec<Bin>,
    bin_capacity: f32,
    best_current: Arc<RwLock<Vec<Bin>>>,
) {
    let lower_bound = simple_lower_bound(&items, bin_capacity).ceil() + filled_bins.len() as f32;
    // If the current possible lower bound is more than the current best, no need to check
    if lower_bound > best_current.read().unwrap().len() as f32 {
        return;
    }
    // If no more values, and solution is better, replace solution
    if items.len() == 0 {
        if filled_bins.len() < best_current.read().unwrap().len() {
            *best_current.write().unwrap() = filled_bins.clone();
            // println!("Updating best found");
            // println!("filled_bins {:?}", filled_bins);
            // println!("Curent best {:?}", best_current);
            // println!("Curent best bins {:?}", best_current.read().unwrap().len());
        }
        return;
    }

    for (item_idx, item) in items.iter().enumerate() {
        // There must be at lease one, we checked for the 0 case
        let mut items_copy = items.clone();
        items_copy.remove(item_idx);
        for (idx, bin) in filled_bins.iter().enumerate() {
            if bin.can_fit(*item) {
                let mut filled_bins_copy = filled_bins.clone();
                let best_current_copy = Arc::clone(&best_current);
                let items_copy_copy = items_copy.clone();
                filled_bins_copy[idx].add(*item);
                rayon::scope(|s| {
                    s.spawn(move |_| {
                        stupid_bin_packing(
                            items_copy_copy,
                            filled_bins_copy,
                            bin_capacity,
                            best_current_copy,
                        )
                    })
                });
            }
        }
        let best_current_copy = Arc::clone(&best_current);
        let mut bin = Bin::new(bin_capacity);
        // Assume item can always fit in new bin
        bin.add(*item);
        //println!("New Bin {:?}", bin);
        let mut filled_bins_copy = filled_bins.clone();
        filled_bins_copy.push(bin);
        rayon::scope(|s| {
            s.spawn(move |_| {
                stupid_bin_packing(
                    items_copy,
                    filled_bins_copy,
                    bin_capacity,
                    best_current_copy,
                )
            })
        });
    }
}

pub fn best_fit_first(items: &Vec<f32>, bin_capacity: f32) -> Vec<Bin> {
    // Should be improved by returning a
    fn find_best_fit(filled_bins: &Vec<Bin>, item: f32) -> Option<usize> {
        let selected = filled_bins
            .into_iter()
            .enumerate()
            .filter(|(_, x)| x.can_fit(item))
            .fold(None, |min, (idx, x)| match min {
                None => Some((idx, x)),
                Some((idy, y)) => Some(if x.residue() < y.residue() {
                    (idx, x)
                } else {
                    (idy, y)
                }),
            });
        match selected {
            Some((idx, _)) => Some(idx),
            None => None,
        }
    }

    let mut filled_bins = Vec::new();

    let mut sorted_items = items.clone();
    sorted_items.sort_by(|a, b| a.partial_cmp(b).unwrap());
    while sorted_items.len() > 0 {
        let largest = sorted_items.pop().unwrap();
        let best_bin_index = find_best_fit(&filled_bins, largest);
        match best_bin_index {
            Some(best_bin_index) => filled_bins[best_bin_index].add(largest),
            None => {
                let mut new_bin = Bin::new(bin_capacity);
                new_bin.add(largest);
                filled_bins.push(new_bin);
            }
        }
    }
    filled_bins
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[test]
    fn test_simple_lower_bound() {
        assert_eq!(
            simple_lower_bound(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0),
            3.0
        )
    }

    #[test]
    fn test_l2_lower_bound_1() {
        assert_eq!(
            l2_lower_bound(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0),
            3.0
        )
    }

    #[test]
    fn test_l2_lower_bound_2() {
        assert_eq!(
            l2_lower_bound(&vec![99.0, 97.0, 94.0, 93.0, 8.0, 5.0, 4.0, 2.0], 100.0),
            5.0
        )
    }

    #[test]
    fn test_best_fit_first() {
        best_fit_first(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0);
    }
    #[test]
    fn test_stupid_bin_packing_very_simple() {
        assert_eq!(stupid_bin_packing_outer(&vec![1.0, 6.0], 7.0).len(), 1);
    }

    #[test]
    fn test_stupid_bin_packing_simple() {
        assert_eq!(
            stupid_bin_packing_outer(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0).len(),
            3
        );
    }

    #[test]
    fn test_stupid_bin_packing_gils1() {
        assert_eq!(
            stupid_bin_packing_outer(
                &vec![25.0, 42.0, 13.0, 31.0, 34.0, 59.0, 13.0, 36.0, 1.0, 61.0],
                64.0,
            )
            .len(),
            6
        );
    }

    #[test]
    fn test_stupid_bin_packing_gils2() {
        assert_eq!(
            stupid_bin_packing_outer(
                &vec![2.0, 63.0, 12.0, 18.0, 34.0, 28.0, 46.0, 51.0, 53.0, 20.0],
                64.0,
            )
            .len(),
            6
        );
    }

    #[test]
    fn test_stupid_bin_packing_gils3() {
        assert_eq!(
            stupid_bin_packing_outer(
                &vec![38.0, 59.0, 53.0, 48.0, 40.0, 55.0, 60.0, 62.0, 3.0, 64.0],
                64.0,
            )
            .len(),
            9
        );
    }

    #[test]
    fn test_stupid_bin_packing_paper() {
        assert_eq!(
            stupid_bin_packing_outer(
                &vec![
                    100.0, 98.0, 96.0, 93.0, 91.0, 87.0, 81.0, 59.0, 58.0, 55.0, 50.0, 43.0, 22.0,
                    21.0, 20.0, 15.0, 14.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 1.0,
                ],
                100.0,
            )
            .len(),
            11
        );
    }

    #[test]
    fn test_stupid_bin_packing_long() {
        assert_eq!(
            stupid_bin_packing_outer(
                &vec![
                    100.0, 98.0, 96.0, 97.0, 93.0, 91.0, 87.0, 83.0, 81.0, 59.0, 58.0, 55.0, 50.0,
                    43.0, 22.0, 21.0, 20.0, 15.0, 14.0, 11.0, 10.0, 8.0, 8.0, 6.0, 5.0, 5.0, 4.0,
                    3.0, 100.0, 98.0, 96.0, 97.0, 93.0, 91.0, 87.0, 83.0, 81.0, 59.0, 58.0, 55.0,
                    50.0, 43.0, 25.0, 22.0, 21.0, 20.0, 15.0, 14.0, 11.0, 10.0, 8.0, 8.0, 6.0, 5.0,
                    5.0, 4.0, 3.0, 1.0,
                ],
                100.0,
            )
            .len(),
            26,
        );
    }

    #[test]
    fn test_stupid_bin_packing_med() {
        assert_eq!(
            stupid_bin_packing_outer(
                &vec![
                    100.0, 98.0, 96.0, 97.0, 93.0, 91.0, 87.0, 83.0, 81.0, 59.0, 58.0, 53.0, 50.0,
                    14.0, 11.0, 10.0, 8.0, 8.0, 6.0, 5.0, 5.0, 4.0, 5.0, 4.0, 3.0, 1.0, 8.0, 9.0,
                    11.0, 14.0, 11.0, 17.0, 1.0, 11.0, 14.0, 11.0, 17.0, 1.0, 11.0, 14.0, 11.0,
                    11.0, 14.0, 11.0, 17.0, 1.0, 11.0, 14.0, 11.0, 17.0, 1.0, 11.0, 14.0, 11.0,
                    11.0, 14.0, 11.0, 17.0, 1.0, 11.0, 14.0, 11.0, 17.0, 1.0, 11.0, 14.0, 11.0,
                    17.0, 1.0, 11.0, 14.0, 11.0, 17.0, 1.0, 17.0, 1.0, 11.0, 14.0, 11.0, 17.0, 1.0,
                ],
                100.0,
            )
            .len(),
            18,
        );
    }

    #[test]
    fn test_stupid_bin_packing_optimal_short() {
        assert_eq!(
            stupid_bin_packing_outer(&vec![5.0, 4.0, 3.0, 3.0, 3.0, 2.0], 10.0).len(),
            2
        );
    }

    #[test]
    fn test_stupid_bin_packing_optimal_long() {
        assert_eq!(
            stupid_bin_packing_outer(
                &vec![5.0, 4.0, 3.0, 3.0, 3.0, 2.0, 5.0, 4.0, 3.0, 3.0, 3.0, 2.0],
                10.0
            )
            .len(),
            4
        );
    }

    #[test]
    fn test_get_all_undominated() {
        get_all_undominated(&vec![1.0, 2.0, 3.0, 4.0], 6.0);
    }

    #[bench]
    fn bench_optimal_short(b: &mut Bencher) {
        b.iter(|| stupid_bin_packing_outer(&vec![5.0, 4.0, 3.0, 3.0, 3.0, 2.0], 10.0));
    }

}
