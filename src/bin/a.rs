#![allow(non_snake_case, unused_macros, clippy::needless_range_loop)]
use itertools::Itertools;
use proconio::{input, marker::Chars};
use rand::{Rng, SeedableRng};

const TIMELIMIT: f64 = 1.935;
const N: usize = 20;
const L: usize = 200;
const DIJ: [(usize, usize); 4] = [(!0, 0), (0, !0), (1, 0), (0, 1)];
const DIR: [char; 4] = ['U', 'L', 'D', 'R'];

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

#[derive(Clone, Debug)]
struct Input {
    s: (usize, usize),
    t: (usize, usize),
    p: f64,
    hs: Vec<Vec<bool>>,
    vs: Vec<Vec<bool>>,
}

impl Input {
    fn can_move(&self, i: usize, j: usize, d: usize) -> bool {
        match d {
            0 => i > 0 && !self.vs[i - 1][j],
            1 => j > 0 && !self.hs[i][j - 1],
            2 => i < N - 1 && !self.vs[i][j],
            3 => j < N - 1 && !self.hs[i][j],
            _ => unreachable!(),
        }
    }
}

fn parse_input() -> Input {
    // fn parse_input() -> (Input, f64, f64) {
    input! {
        s: (usize, usize),
        t: (usize, usize),
        p: f64,
        hs: [Chars; N],
        vs: [Chars; N - 1],
        // s_temp: f64,
        // e_temp: f64,
    }
    let hs = hs
        .into_iter()
        .map(|h| h.into_iter().map(|a| a == '1').collect())
        .collect();
    let vs = vs
        .into_iter()
        .map(|v| v.into_iter().map(|a| a == '1').collect())
        .collect();
    Input { s, t, p, hs, vs }
    // (Input { s, t, p, hs, vs }, s_temp, e_temp)
}

fn main() {
    let mut timer = Timer::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let input = parse_input();
    // let (input, s_temp, e_temp) = parse_input();
    let mut output = find_path(&input, &mut rng);
    // let mut output = find_path(&input);
    annealing(&input, &mut output, &mut timer, &mut rng);
    // annealing(&input, &mut output, &mut timer, &mut rng, s_temp, e_temp);
    let answer = output.iter().collect::<String>();
    println!("{}", answer);
    // eprintln!("{}", compute_score(&input, &output).0);
}

fn find_path(input: &Input, rng: &mut rand_chacha::ChaCha20Rng) -> Vec<char> {
    // fn find_path(input: &Input) -> Vec<char> {
    let mut heap = std::collections::BinaryHeap::new();
    let mut dist = vec![vec![i32::max_value(); N]; N];
    let mut prev = vec![vec![!0; N]; N];
    let mut weight = vec![vec![0; N]; N];
    for line in weight.iter_mut() {
        for w in line.iter_mut() {
            *w = rng.gen_range(1i32, 1001i32);
        }
    }
    dist[input.s.0][input.s.1] = 0;
    heap.push((0, input.s));
    while let Some((cost, (i, j))) = heap.pop() {
        let cost = -cost;
        if cost > dist[i][j] {
            continue;
        }
        for (d, &(di, dj)) in DIJ.iter().enumerate() {
            if !input.can_move(i, j, d) {
                continue;
            }
            let ni = i + di;
            let nj = j + dj;
            if dist[ni][nj] > cost + weight[i][j] {
                dist[ni][nj] = cost + weight[i][j];
                prev[ni][nj] = d;
                heap.push((-(cost + weight[i][j]), (ni, nj)));
            }
        }
    }

    // 経路復元
    let mut output = vec![];
    let mut p = input.t;
    while p != input.s {
        let d = prev[p.0][p.1];
        output.push(DIR[d]);
        let q = (p.0 - DIJ[d].0, p.1 - DIJ[d].1);
        p = q;
    }
    output.reverse();
    // ダブらせ
    // while output.len() < L {
    //     let duplicated_index = rng.gen_range(0, output.len());
    //     let element = output[duplicated_index];
    //     output.insert(duplicated_index, element);
    // }
    // Lに足りない分適当に足す
    output.append(
        &mut DIR
            .iter()
            .cycle()
            .take(L - output.len())
            .collect::<String>()
            .chars()
            .collect_vec(),
    );
    output
}

fn annealing(
    input: &Input,
    output: &mut Vec<char>,
    timer: &mut Timer,
    rng: &mut rand_chacha::ChaCha20Rng,
    // s_temp: f64,
    // e_temp: f64,
) -> i64 {
    const T0: f64 = 10.0;
    const T1: f64 = 0.00001;
    let mut temp = T0;
    // let mut temp = s_temp;
    let mut prob;

    let mut count = 0;
    let mut now_score = compute_score(input, output).0;

    let mut best_score = now_score;
    let mut best_output = output.clone();
    const NEIGH_COUNT: i32 = 6;
    loop {
        if count >= 100 {
            let passed = timer.get_time() / TIMELIMIT;
            if passed >= 1.0 {
                break;
            }
            // eprintln!("{} {}", temp, now_score);
            temp = T0.powf(1.0 - passed) * T1.powf(passed);
            // temp = s_temp.powf(1.0 - passed) * e_temp.powf(passed);
            count = 0;
        }
        count += 1;

        let mut new_out = output.clone();
        let neigh_type = rng.gen_range(0, NEIGH_COUNT);
        match neigh_type {
            0 => {
                // swap
                let swap_index1 = rng.gen_range(0, new_out.len());
                let swap_index2 = rng.gen_range(0, new_out.len());
                let out1 = new_out[swap_index1];
                let out2 = new_out[swap_index2];
                new_out[swap_index1] = out2;
                new_out[swap_index2] = out1;
            }
            1 => {
                if new_out.len() == L {
                    continue;
                }
                // insert
                let insert_index = rng.gen_range(0, new_out.len());
                let insert_dir = rng.gen_range(0, 4);
                new_out.insert(insert_index, DIR[insert_dir]);
            }
            2 => {
                if new_out.len() < 2 {
                    continue;
                }
                // remove
                let remove_index = rng.gen_range(0, new_out.len());
                new_out.remove(remove_index);
            }
            3 => {
                if new_out.len() == L {
                    continue;
                }
                // duplicate
                let dup_index = rng.gen_range(0, new_out.len());
                new_out.insert(dup_index, new_out[dup_index]);
            }
            4 => {
                // move
                let src_index = rng.gen_range(0, new_out.len());
                let move_dir = new_out[src_index];
                new_out.remove(src_index);
                let dst_index = rng.gen_range(0, new_out.len());
                new_out.insert(dst_index, move_dir);
            }
            5 => {
                // update
                let update_index = rng.gen_range(0, new_out.len());
                let new_dir = rng.gen_range(0, 4);
                new_out[update_index] = DIR[new_dir];
            }
            _ => unreachable!(),
        }
        let new_score = compute_score(input, &new_out).0;
        prob = f64::exp((new_score - now_score) as f64 / temp);
        if now_score < new_score || rng.gen_bool(prob) {
            now_score = new_score;
            *output = new_out;
        }

        if best_score < now_score {
            best_score = now_score;
            best_output = output.clone();
        }
    }
    eprintln!("{}", best_score);
    *output = best_output;
    best_score
}

fn compute_score(input: &Input, out: &[char]) -> (i64, String, Vec<Vec<f64>>) {
    let mut crt = mat![0.0; N; N];
    crt[input.s.0][input.s.1] = 1.0;
    let mut sum = 0.0;
    let mut goal = 0.0;
    for (i, t) in out.iter().enumerate() {
        if let Some(d) = DIR.iter().position(|&c| c == *t) {
            let mut next = mat![0.0; N; N];
            for i in 0..N {
                for j in 0..N {
                    if crt[i][j] > 0.0 {
                        if input.can_move(i, j, d) {
                            let i2 = i + DIJ[d].0;
                            let j2 = j + DIJ[d].1;
                            next[i2][j2] += crt[i][j] * (1.0 - input.p);
                            next[i][j] += crt[i][j] * input.p;
                        } else {
                            next[i][j] += crt[i][j];
                        }
                    }
                }
            }
            crt = next;
            sum += crt[input.t.0][input.t.1] * (2 * L - i) as f64;
            goal += crt[input.t.0][input.t.1];
            crt[input.t.0][input.t.1] = 0.0;
        } else {
            return (0, format!("illegal char: {}", *t), crt);
        }
    }
    crt[input.t.0][input.t.1] = goal;
    (
        (1e8 * sum / (2 * L) as f64).round() as i64,
        String::new(),
        crt,
    )
}

fn get_time() -> f64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9
}

struct Timer {
    start_time: f64,
}

impl Timer {
    fn new() -> Timer {
        Timer {
            start_time: get_time(),
        }
    }

    fn get_time(&self) -> f64 {
        get_time() - self.start_time
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.start_time = 0.0;
    }
}
