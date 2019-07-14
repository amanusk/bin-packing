#[derive(Clone, PartialOrd)]
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

    pub fn get_capacity(&self) -> f32 {
        return self.capacity;
    }

    pub fn get_total(&self) -> f32 {
        return self.total;
    }

    pub fn get_residue(&self) -> f32 {
        return self.residue;
    }

    pub fn get_items(&self) -> &Vec<f32> {
        return &self.items;
    }
}

fn simple_lower_bound(items: Vec<f32>, bin_capacity: f32) -> f32 {
    let sum_items = items.iter().fold(0.0, |sum, x| sum + x);
    sum_items / (bin_capacity)
}

fn l2_lower_bound(items: Vec<f32>, bin_capacity: f32) -> f32 {
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

fn best_fit_first(items: Vec<f32>, bin_capacity: f32) -> Vec<Vec<f32>> {
    // Find the index of the best fitting bin, -1 if there is no good fit
    fn find_best_fit_index(filled_bins: &Vec<Vec<f32>>, item: f32, bin_capacity: f32) -> i32 {
        if filled_bins.len() == 0 {
            return -1;
        }
        let mut best_residue = bin_capacity;
        let mut best_bin_index = filled_bins.len();
        for (bin_idx, bin) in filled_bins.iter().enumerate() {
            let total: f32 = bin.clone().into_iter().sum();
            if item + total > bin_capacity {
                continue;
            } else {
                if bin_capacity - total < best_residue {
                    best_residue = bin_capacity - total;
                    best_bin_index = bin_idx;
                }
            }
        }
        if best_bin_index < filled_bins.len() {
            best_bin_index as i32
        } else {
            -1
        }
    }

    // Should be improved by returning a
    // fn find_best_fit(
    //     filled_bins: &Vec<Vec<f32>>,
    //     item: f32,
    //     bin_capacity: f32,
    // ) -> Option<&Vec<f32>> {
    //     let possible_bins = filled_bins
    //         .into_iter()
    //         .filter(|bin: Vec<f32>| bin.iter().sum() + item <= bin_capacity);
    //     let selected = possible_bins.into_iter().fold(None, |min, x| match min {
    //         None => Some(x),
    //         Some(y) => Some(
    //             if bin_capacity - x.iter().sum() < bin_capacity - y.iter().sum() {
    //                 x
    //             } else {
    //                 y
    //             },
    //         ),
    //     });
    //     selected
    // }

    let mut filled_bins = Vec::new();

    let mut sorted_items = items.clone();
    sorted_items.sort_by(|a, b| a.partial_cmp(b).unwrap());
    while sorted_items.len() > 0 {
        let largest = sorted_items.pop().unwrap();
        let best_bin_index = find_best_fit_index(&filled_bins, largest, bin_capacity);
        if best_bin_index >= 0 {
            filled_bins[best_bin_index as usize].push(largest);
        } else {
            filled_bins.push(vec![largest])
        }
    }
    filled_bins
}

#[cfg(test)]
mod tests {
    use super::{best_fit_first, l2_lower_bound, simple_lower_bound};

    #[test]
    fn test_simple_lower_bound() {
        assert_eq!(
            simple_lower_bound(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0),
            3.0
        )
    }

    #[test]
    fn test_l2_lower_bound_1() {
        assert_eq!(l2_lower_bound(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0), 3.0)
    }

    #[test]
    fn test_l2_lower_bound_2() {
        assert_eq!(
            l2_lower_bound(vec![99.0, 97.0, 94.0, 93.0, 8.0, 5.0, 4.0, 2.0], 100.0),
            5.0
        )
    }

    #[test]
    fn test_best_fit_first() {
        best_fit_first(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0);
    }

}
