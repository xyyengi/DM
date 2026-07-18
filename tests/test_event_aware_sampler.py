import unittest

from src.training.event_aware_sampler import EventAwareBatchSampler


def synthetic_pools():
    pools = {}
    for event_type, offset in (("high_load", 0), ("low_wind", 20)):
        pools[event_type] = {}
        for event_number in range(5):
            event_id = f"{event_type}_{event_number}"
            pools[event_type][event_id] = {
                "0-24h": [offset + event_number],
                "24-48h": [offset + event_number + 5],
            }
    return pools


class EventAwareBatchSamplerTests(unittest.TestCase):
    def make_sampler(self, seed=7):
        return EventAwareBatchSampler(
            dataset_size=100,
            event_pools=synthetic_pools(),
            batch_size=16,
            event_fraction=0.25,
            seed=seed,
            max_draws_per_event_per_epoch=4,
        )

    def test_epoch_size_and_batch_sizes(self):
        sampler = self.make_sampler()

        batches = list(sampler)

        self.assertEqual(sum(map(len, batches)), 100)
        self.assertEqual([len(batch) for batch in batches], [16] * 6 + [4])
        self.assertEqual(sampler.last_epoch_stats["targeted_event_draws"], 25)
        self.assertAlmostEqual(sampler.last_epoch_stats["targeted_event_fraction"], 0.25)

    def test_hierarchy_balances_types_and_caps_event_ids(self):
        sampler = self.make_sampler()

        list(sampler)
        type_counts = sampler.last_epoch_stats["targeted_draws_by_event_type"]

        self.assertLessEqual(max(type_counts.values()) - min(type_counts.values()), 1)
        self.assertLessEqual(
            sampler.last_epoch_stats["max_targeted_draws_for_one_event_id"], 4
        )
        self.assertEqual(
            sum(sampler.last_epoch_stats["targeted_draws_by_lead_group"].values()), 25
        )
        lead_counts = sampler.last_epoch_stats["targeted_draws_by_lead_group"]
        self.assertLessEqual(max(lead_counts.values()) - min(lead_counts.values()), 6)

    def test_same_seed_and_epoch_are_reproducible(self):
        first = self.make_sampler(seed=9)
        second = self.make_sampler(seed=9)
        first.set_epoch(3)
        second.set_epoch(3)

        self.assertEqual(list(first), list(second))

    def test_different_epochs_change_batches(self):
        sampler = self.make_sampler(seed=9)
        sampler.set_epoch(0)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
