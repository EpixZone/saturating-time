//! Internal parts used for sealing.
//!
//! This module primarily consists of the internal [`SaturatingTime`] trait, an
//! unstable abstraction used internally to implement the main logic behind
//! this.
//!
//! Normal users should not be using this.

use std::{
    cmp,
    sync::LazyLock,
    time::{Duration, Instant, SystemTime},
};

/// The maximum value of [`SystemTime`] for this platform.
static MAX_SYSTEM_TIME: LazyLock<SystemTime> = LazyLock::new(find_max);

/// The minimum value of [`SystemTime`] for this platform.
static MIN_SYSTEM_TIME: LazyLock<SystemTime> = LazyLock::new(find_min);

/// The maximum value of [`Instant`] for this platform.
static MAX_INSTANT: LazyLock<Instant> = LazyLock::new(find_max);

/// The minimum value of [`Instant`] for this platform.
static MIN_INSTANT: LazyLock<Instant> = LazyLock::new(find_min);

/// An internal trait implementing the actual magic behind this.
pub trait SaturatingTime: Sized + Copy + PartialEq {
    /// Anchor method to obtain an instance of this type.
    fn anchor() -> Self;

    /// Returns the maximum value of this type.
    fn max_value() -> Self;

    /// Returns the minimum value of this type.
    fn min_value() -> Self;

    /// Performs a checked addition on this type.
    fn checked_add(&self, duration: Duration) -> Option<Self>;

    /// Performs a checked subtraction on this type.
    fn checked_sub(&self, duration: Duration) -> Option<Self>;

    /// Performs a checked time delta on this type.
    fn checked_duration_since(&self, earlier: Self) -> Option<Duration>;
}

impl SaturatingTime for SystemTime {
    fn anchor() -> Self {
        Self::UNIX_EPOCH
    }

    fn max_value() -> Self {
        *MAX_SYSTEM_TIME
    }

    fn min_value() -> Self {
        *MIN_SYSTEM_TIME
    }

    fn checked_add(&self, duration: Duration) -> Option<Self> {
        Self::checked_add(self, duration)
    }

    fn checked_sub(&self, duration: Duration) -> Option<Self> {
        Self::checked_sub(self, duration)
    }

    fn checked_duration_since(&self, earlier: Self) -> Option<Duration> {
        Self::duration_since(self, earlier).ok()
    }
}

impl SaturatingTime for Instant {
    fn anchor() -> Self {
        Self::now()
    }

    fn max_value() -> Self {
        *MAX_INSTANT
    }

    fn min_value() -> Self {
        *MIN_INSTANT
    }

    fn checked_add(&self, duration: Duration) -> Option<Self> {
        Self::checked_add(self, duration)
    }

    fn checked_sub(&self, duration: Duration) -> Option<Self> {
        Self::checked_sub(self, duration)
    }

    /// DO NOT USE!
    ///
    /// Instead, override the top-level provided implementation with the already
    /// existing [`Instant::saturating_duration_since()`].
    fn checked_duration_since(&self, _earlier: Self) -> Option<Duration> {
        unreachable!()
    }
}

/// Finds the value for [`SaturatingTime::max_value()`].
fn find_max<T: SaturatingTime>() -> T {
    find_limit(T::checked_add)
}

/// Finds the value for [`SaturatingTime::min_value()`].
fn find_min<T: SaturatingTime>() -> T {
    find_limit(T::checked_sub)
}

/// Internal algorithm of [`find_max()`] and [`find_min()`].
///
/// It works by performing `f` with a very large [`Duration`] onto
/// [`SaturatingTime::anchor()`] until this call returns [`None`], in which case
/// this [`Duration`] gets halved.  This process is repeated until `f` returns
/// [`None`] and the [`Duration`] has reached 1ns.
///
/// # Algorithm
///
/// 1. Set `step` to `INITIAL_STEP` and `res` to [`SaturatingTime::anchor()`].
/// 2. Call `f(&res, step)`.
///     1. If [`Some`] with a *different* value, set `res` to it and continue.
///     2. Otherwise (`None`, or [`Some`] that did not move `res`): if
///        `step == 1ns`, return `res`; else halve `step` and continue.
///
/// # Why "`Some` that did not move `res`" must also shrink the step
///
/// The step is only shrunk when `f` fails to make progress. Originally that was
/// keyed solely on `f` returning [`None`] (an overflow of the representable
/// range). That assumes 1ns granularity: every `step >= 1ns` either advances
/// `res` or overflows.
///
/// On platforms whose clock is coarser than 1ns this does not hold. Windows
/// `SystemTime` is a `FILETIME` with 100ns granularity, so any `step` below
/// 100ns rounds to a no-op: `f` returns `Some(res)` (unchanged) forever, `step`
/// is never shrunk, `step == 1ns` is never reached, and `find_limit` spins
/// forever. Treating "`Some` but unchanged" as a non-progressing step fixes the
/// hang and still converges to the true limit (to within the clock's
/// granularity, which is the best any value of this type can represent).
fn find_limit<T, F>(f: F) -> T
where
    T: SaturatingTime,
    F: Fn(&T, Duration) -> Option<T>,
{
    const INITIAL_STEP: Duration = Duration::new(1_000_000_000_000_000_000, 0);
    const ONE_NS: Duration = Duration::new(0, 1);

    // (1) Set step to INITIAL_STEP and res to T::anchor().
    let mut step = INITIAL_STEP;
    let mut res = T::anchor();

    loop {
        // (2) Call f().
        match f(&res, step) {
            // (2.1) Progress: a different value, so advance and keep this step.
            Some(next) if next != res => res = next,
            // (2.2) No progress: `f` overflowed (None), or the step was too
            // small to change `res` on this platform's clock. This step size is
            // exhausted. Stop at 1ns, else halve and retry.
            _ => {
                if step == ONE_NS {
                    return res;
                }
                step = cmp::max(ONE_NS, step / 2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fmt::Debug,
        ops::{Add, Sub},
    };

    use super::*;

    /// Checks whether the minimum and maximum values are correct.
    fn min_max<T>()
    where
        T: SaturatingTime
            + PartialEq
            + Debug
            + Add<Duration, Output = T>
            + Sub<Duration, Output = T>,
    {
        assert_eq!(
            T::max_value().checked_add(Duration::ZERO),
            Some(T::max_value())
        );
        assert_eq!(T::max_value().checked_add(Duration::new(0, 1)), None);
        assert_eq!(
            T::max_value().checked_sub(Duration::ZERO),
            Some(T::max_value())
        );
        assert_eq!(
            T::max_value().checked_sub(Duration::new(0, 1)),
            Some(T::max_value() - Duration::new(0, 1))
        );

        assert_eq!(
            T::min_value().checked_sub(Duration::ZERO),
            Some(T::min_value())
        );
        assert_eq!(T::min_value().checked_sub(Duration::new(0, 1)), None);
        assert_eq!(
            T::min_value().checked_add(Duration::ZERO),
            Some(T::min_value())
        );
        assert_eq!(
            T::min_value().checked_add(Duration::new(0, 1)),
            Some(T::min_value() + Duration::new(0, 1))
        );
    }

    /// Verifies [`SystemTime::min_value()`] and [`SystemTime::max_value()`] are
    /// correct.
    ///
    /// Unix only: `min_max` asserts that adding/subtracting 1ns at the limit
    /// overflows. `SystemTime` on Windows is a `FILETIME` with 100ns
    /// granularity, so a 1ns step at the limit is a no-op rather than an
    /// overflow, and this assertion does not hold there.
    #[cfg(target_family = "unix")]
    #[test]
    fn system_time_min_max() {
        min_max::<SystemTime>();
    }

    /// Verifies [`Instant::min_value()`] and [`Instant::max_value()`] are
    /// correct.
    #[test]
    fn instant_min_max() {
        min_max::<Instant>();
    }

    /// Verifies [`SystemTime::min_value()`] and [`SystemTime::max_value()`] are
    /// correct on Unix systems.
    #[cfg(target_family = "unix")]
    #[test]
    fn system_time_min_max_unix() {
        assert_eq!(
            SystemTime::max_value(),
            SystemTime::UNIX_EPOCH + Duration::new(i64::MAX as u64, 999_999_999)
        );
        assert_eq!(
            SystemTime::min_value(),
            SystemTime::UNIX_EPOCH - Duration::new(i64::MAX as u64 + 1, 0)
        );
    }

    /// Verifies that [`Instant::min_value()`] and [`Instant::max_value()`] are
    /// correct on Unix systems.
    #[test]
    #[cfg(target_family = "unix")]
    fn instant_min_max_unix() {
        // Using format is not nice but I cannot see a better way for now.
        assert_eq!(
            format!("{:?}", Instant::max_value()),
            format!(
                "Instant {{ tv_sec: {}, tv_nsec: {} }}",
                i64::MAX,
                999_999_999
            )
        );

        assert_eq!(
            format!("{:?}", Instant::min_value()),
            format!("Instant {{ tv_sec: {}, tv_nsec: {} }}", i64::MIN, 0)
        );
    }
}
