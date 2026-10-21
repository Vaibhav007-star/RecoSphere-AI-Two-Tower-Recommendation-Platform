"""
Baseline Recommender Models for RecoSphere AI.
High-performance implementations:
1. Global Popularity Baseline
2. Category Affinity Content-Based Baseline
3. Random Recommender
"""

from typing import List, Dict, Set, Optional
import numpy as np
import pandas as pd


class PopularityRecommender:
    """Recommends globally most popular items based on training interaction weights."""

    def __init__(self):
        self.popular_items: List[int] = []
        self.item_scores: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame):
        """Fits popularity distribution using weighted interaction counts."""
        weight_col = "weight" if "weight" in train_df.columns else None
        if weight_col:
            grouped = train_df.groupby("itemid")["weight"].sum()
        else:
            grouped = train_df.groupby("itemid").size()

        sorted_items = grouped.sort_values(ascending=False)
        self.popular_items = [int(i) for i in sorted_items.index.tolist()]
        max_score = float(sorted_items.max()) if len(sorted_items) > 0 else 1.0
        self.item_scores = {int(i): float(s / max_score) for i, s in sorted_items.items()}

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: Optional[Set[int]] = None,
    ) -> List[int]:
        """Returns top_k most popular items excluding already seen items."""
        exclude = exclude_seen or set()
        recommendations = []
        for item in self.popular_items:
            if item not in exclude:
                recommendations.append(item)
            if len(recommendations) >= top_k:
                break
        return recommendations

    def get_score(self, item_id: int) -> float:
        return self.item_scores.get(item_id, 0.0)


class CategoryAffinityRecommender:
    """Content-based baseline recommending popular items matching user historical category affinities."""

    def __init__(self):
        self.user_fav_categories: Dict[int, List[int]] = {}
        self.category_popular_items: Dict[int, List[int]] = {}
        self.fallback_popularity = PopularityRecommender()
        self.item_to_category: Dict[int, int] = {}

    def fit(self, train_df: pd.DataFrame, catalog_df: pd.DataFrame, eval_users: Optional[List[int]] = None):
        """Fits per-category popular items and user category preferences."""
        self.fallback_popularity.fit(train_df)

        if "categoryid" in catalog_df.columns:
            valid_cats = catalog_df.dropna(subset=["categoryid"])
            self.item_to_category = dict(zip(valid_cats["itemid"].astype(int).values, valid_cats["categoryid"].astype(int).values))

        # 1. Per-category popularity (only 1,218 categories - instant vectorized calculation)
        train_items = train_df["itemid"].values
        mapped_cats = [self.item_to_category.get(int(i)) for i in train_items]
        cat_df = pd.DataFrame({"itemid": train_items, "categoryid": mapped_cats}).dropna()
        
        cat_item_counts = cat_df.groupby(["categoryid", "itemid"]).size().reset_index(name="count")
        cat_item_counts = cat_item_counts.sort_values(["categoryid", "count"], ascending=[True, False])
        
        for cat_id, grp in cat_item_counts.groupby("categoryid"):
            self.category_popular_items[int(cat_id)] = grp["itemid"].head(50).tolist()

        # 2. User category preferences (only compute for evaluated/active users or top users)
        if eval_users is not None:
            sub_train = train_df[train_df["visitorid"].isin(eval_users)]
        else:
            # Default to top 50,000 active users
            top_u = train_df["visitorid"].value_counts().head(50000).index
            sub_train = train_df[train_df["visitorid"].isin(top_u)]

        sub_items = sub_train["itemid"].values
        sub_cats = [self.item_to_category.get(int(i)) for i in sub_items]
        user_cat_df = pd.DataFrame({"visitorid": sub_train["visitorid"].values, "categoryid": sub_cats}).dropna()

        # Fast group by visitor and category
        u_counts = user_cat_df.groupby(["visitorid", "categoryid"]).size().reset_index(name="cnt")
        u_counts = u_counts.sort_values(["visitorid", "cnt"], ascending=[True, False])
        top_cats = u_counts.groupby("visitorid")["categoryid"].head(3)
        top_u_df = u_counts.loc[top_cats.index]

        # Vectorized aggregation using python dictionary build
        user_cats_dict: Dict[int, List[int]] = {}
        for uid, cid in zip(top_u_df["visitorid"].values, top_u_df["categoryid"].values):
            uid_int = int(uid)
            if uid_int not in user_cats_dict:
                user_cats_dict[uid_int] = []
            user_cats_dict[uid_int].append(int(cid))

        self.user_fav_categories = user_cats_dict

    def get_user_categories(self, user_id: int) -> List[int]:
        return self.user_fav_categories.get(user_id, [])

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: Optional[Set[int]] = None,
    ) -> List[int]:
        exclude = exclude_seen or set()
        user_cats = self.user_fav_categories.get(user_id, [])

        if not user_cats:
            return self.fallback_popularity.recommend(user_id, top_k, exclude)

        recommendations = []
        cat_lists = [self.category_popular_items.get(c, []) for c in user_cats]
        max_len = max((len(l) for l in cat_lists), default=0)

        for idx in range(max_len):
            for c_list in cat_lists:
                if idx < len(c_list):
                    item = c_list[idx]
                    if item not in exclude and item not in recommendations:
                        recommendations.append(item)
                    if len(recommendations) >= top_k:
                        return recommendations

        if len(recommendations) < top_k:
            fallback = self.fallback_popularity.recommend(user_id, top_k * 2, exclude)
            for itm in fallback:
                if itm not in recommendations:
                    recommendations.append(itm)
                if len(recommendations) >= top_k:
                    break

        return recommendations[:top_k]

