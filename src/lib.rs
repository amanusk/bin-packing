use std::sync::{Arc, Mutex};

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

fn simple_lower_bound(items: &Vec<f32>, bin_capacity: f32) -> f32 {
    let sum_items: f32 = items.iter().sum();
    sum_items / bin_capacity
}

fn stupid_bin_packing_outer(items: Vec<f32>, bin_capacity: f32) -> Vec<Bin> {
    // Set upper bound to the number of items
    let best_current = Arc::new(Mutex::new(Vec::new()));
    for item in items.clone() {
        let mut bin = Bin::new(bin_capacity);
        bin.add(item);
        let best_inner = Arc::clone(&best_current);
        best_inner.lock().unwrap().push(bin);
    }
    println!("Starting from {:?}", best_current.clone());
    stupid_bin_packing(
        items.clone(),
        Vec::new(),
        bin_capacity,
        best_current.clone(),
    );
    println!("Best found {:?}", best_current);
    best_current.clone().lock().unwrap().clone()
}

fn stupid_bin_packing(
    items: Vec<f32>,
    filled_bins: Vec<Bin>,
    bin_capacity: f32,
    best_current: Arc<Mutex<Vec<Bin>>>,
) {
    println!("Currnt best {:?}", best_current.clone());
    println!("Left items {:?}", items);
    println!("filled_bins {:?}", filled_bins);
    let simple_lower_bound = simple_lower_bound(&items, bin_capacity) + filled_bins.len() as f32;
    // If the current possible lower bound is more than the current best, no need to check
    if simple_lower_bound > best_current.clone().lock().unwrap().len() as f32 {
        return;
    }
    // If no more values, and solution is better, replace solution
    if items.len() == 0 {
        if filled_bins.len() < best_current.clone().lock().unwrap().len() {
            *best_current.clone().lock().unwrap() = filled_bins.clone();
            println!("Updating best found");
            println!("filled_bins {:?}", filled_bins);
            println!("Curent best {:?}", best_current);
        }
        return;
    }
    // There must be at lease one, we checked for the 0 case
    let mut items = items.clone();
    let item = items.pop().unwrap();
    for (idx, bin) in filled_bins.iter().enumerate() {
        if bin.can_fit(item) {
            let mut copy = filled_bins.clone();
            copy[idx].add(item);
            stupid_bin_packing(items.clone(), copy, bin_capacity, best_current.clone());
        }
    }
    let mut bin = Bin::new(bin_capacity);
    // Assume item can always fit in new bin
    bin.add(item);
    println!("New Bin {:?}", bin);
    let mut filled_bins_copy = filled_bins.clone();
    filled_bins_copy.push(bin);
    stupid_bin_packing(
        items.clone(),
        filled_bins_copy,
        bin_capacity,
        best_current.clone(),
    );
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

fn best_fit_first(items: Vec<f32>, bin_capacity: f32) -> Vec<Bin> {
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
    println!("filled bins {:?}", filled_bins);
    filled_bins
}

#[cfg(test)]
mod tests {
    use super::{best_fit_first, l2_lower_bound, simple_lower_bound, stupid_bin_packing_outer};

    #[test]
    fn test_simple_lower_bound() {
        assert_eq!(
            simple_lower_bound(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0),
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

    #[test]
    fn test_stupid_bin_packing() {
        stupid_bin_packing_outer(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7.0);
    }

}
