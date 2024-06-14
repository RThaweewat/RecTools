"""Latest model."""

import typing as tp
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rectools import Columns, InternalIds
from rectools.dataset import Dataset
from rectools.types import InternalIdsArray
from rectools.utils import fast_isin_for_sorted_test_elements

from .base import FixedColdRecoModelMixin, ModelBase, Scores, ScoresArray
from .utils import get_viewed_item_ids

class LatestModel(FixedColdRecoModelMixin, ModelBase):
    """
    Model generating recommendations based on the most recent interactions.

    Parameters
    ----------
    period : timedelta, optional, default ``None``
        Period before last interaction to consider interactions for recommendations.
        Either `period` or `begin_from` can be set at once.
        If both are ``None`` all interactions will be used.
    begin_from : datetime, optional, default ``None``
        Exact datetime to consider interactions from for recommendations.
        Either `period` or `begin_from` can be set at once.
        If both are ``None`` all interactions will be used.
    add_cold : bool, default ``False``
        If ``True`` cold items will be added to the end of the list and can be recommended.
        Item is cold if it's not present in interactions at all (but present in id map)
        or not present in last interactions defined by either `period` or `begin_from` arguments.
        Order of cold items is unpredictable.
        Cold items score will be equal to ``0``.
    inverse : bool, default ``False``
        If ``True`` least recently interacted items will be selected.
    verbose : int, default ``0``
        Degree of verbose output. If ``0``, no output will be provided.
    """

    recommends_for_warm = False
    recommends_for_cold = True

    def __init__(
        self,
        period: tp.Optional[timedelta] = None,
        begin_from: tp.Optional[datetime] = None,
        add_cold: bool = False,
        inverse: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        if period is not None and begin_from is not None:
            raise ValueError("Only one of `period` and `begin_from` can be set")
        self.period = period
        self.begin_from = begin_from

        self.add_cold = add_cold
        self.inverse = inverse

        self.latest_list: tp.Tuple[InternalIdsArray, ScoresArray]

    def _filter_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if self.begin_from is not None:
            interactions = interactions.loc[interactions[Columns.Datetime] >= self.begin_from]
        elif self.period is not None:
            begin_from = interactions[Columns.Datetime].max() - self.period
            interactions = interactions.loc[interactions[Columns.Datetime] >= begin_from]
        return interactions

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        interactions = self._filter_interactions(dataset.interactions.df)
        interactions = interactions.sort_values(by=Columns.Datetime, ascending=not self.inverse)
        items = interactions[Columns.Item].unique()
        scores = np.arange(len(items), 0, -1).astype(float) if not self.inverse else np.arange(1, len(items) + 1).astype(float)

        if self.add_cold:
            cold_items = np.setdiff1d(dataset.item_id_map.internal_ids, items)
            items = np.concatenate((items, cold_items))
            scores = np.concatenate((scores, np.zeros(cold_items.size)))

        self.latest_list = (items, scores)

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        latest_list = self._get_filtered_latest_list(sorted_item_ids_to_recommend)

        if filter_viewed:
            user_items = dataset.get_user_item_matrix(include_weights=False)

        all_user_ids = []
        all_reco_ids: tp.List[int] = []
        all_scores: tp.List[float] = []
        for user_id in tqdm(user_ids, disable=self.verbose == 0):
            if filter_viewed:
                sorted_blacklist = get_viewed_item_ids(user_items, user_id)
            else:
                sorted_blacklist = None
            reco_ids, reco_scores = self._recommend_for_user(k, latest_list, sorted_blacklist)
            all_user_ids.extend([user_id] * len(reco_ids))
            all_reco_ids.extend(reco_ids)
            all_scores.extend(reco_scores)

        return all_user_ids, all_reco_ids, all_scores

    @classmethod
    def _recommend_for_user(
        cls,
        k: int,
        latest_list: tp.Tuple[InternalIdsArray, ScoresArray],
        sorted_blacklist: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, Scores]:
        if sorted_blacklist is not None:
            n_items = k + sorted_blacklist.size
        else:
            n_items = k

        reco = latest_list[0][:n_items]
        scores = latest_list[1][:n_items]

        if sorted_blacklist is not None:
            valid_mask = fast_isin_for_sorted_test_elements(reco, sorted_blacklist, invert=True)
            reco = reco[valid_mask][:k]
            scores = scores[valid_mask][:k]

        return reco, scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        _, single_reco, single_scores = self._recommend_u2i(
            user_ids=dataset.user_id_map.internal_ids[:1],
            dataset=dataset,
            k=k,
            filter_viewed=False,
            sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
        )

        n_targets = len(target_ids)
        n_reco_per_target = len(single_reco)

        all_target_ids = np.repeat(target_ids, n_reco_per_target)
        all_reco_ids = np.tile(single_reco, n_targets)
        all_scores = np.tile(single_scores, n_targets)
        return all_target_ids, all_reco_ids, all_scores

    def _get_filtered_latest_list(
        self, sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray]
    ) -> tp.Tuple[InternalIdsArray, ScoresArray]:
        latest_list = self.latest_list
        if sorted_item_ids_to_recommend is not None:
            valid_items_mask = fast_isin_for_sorted_test_elements(latest_list[0], sorted_item_ids_to_recommend)
            latest_list = (latest_list[0][valid_items_mask], latest_list[1][valid_items_mask])
        return latest_list

    def _get_cold_reco(
        self, dataset: Dataset, k: int, sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray]
    ) -> tp.Tuple[InternalIds, Scores]:
        latest_list = self._get_filtered_latest_list(sorted_item_ids_to_recommend)
        reco_ids = latest_list[0][:k]
        scores = latest_list[1][:k]
        return reco_ids, scores
