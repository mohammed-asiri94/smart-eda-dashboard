"""User-controlled data-cleaning page."""

import numpy as np
import streamlit as st

from modules.cleaning import clean_dataset, create_multiple_imputed_zip


def render_cleaning_page(
    df,
    numeric_cols,
    categorical_cols,
    constant_cols,
    possible_id_cols,
    total_missing_cells,
    duplicate_rows,
):
    """Render cleaning controls; changes occur only after an explicit user action."""
    st.markdown("## Data Cleaning and Imputation")
    st.warning(
        "Cleaning is applied to a **copy** of the uploaded data. "
        "The original is never modified."
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Rows (before)", f"{df.shape[0]:,}")
    b2.metric("Columns (before)", f"{df.shape[1]:,}")
    b3.metric("Missing (before)", f"{total_missing_cells:,}")
    b4.metric("Duplicates (before)", f"{duplicate_rows:,}")

    st.markdown("---")
    st.markdown("### Basic Cleaning")
    cl1, cl2, cl3 = st.columns(3)
    remove_dups  = cl1.checkbox("Remove duplicate rows", value=True)
    remove_const = cl2.checkbox("Remove constant columns", value=True)
    remove_ids   = cl3.checkbox("Remove possible ID columns", value=False)
    drop_miss_thresh = st.slider(
        "Drop columns with missing % above:", 0, 100, 100, 5,
        help="Set to 100 to disable.",
    )

    st.markdown("---")
    st.markdown("### Missing Value Imputation")
    group_options = ["None"] + categorical_cols
    group_col = st.selectbox("Grouping column (optional)", group_options)
    imp1, imp2 = st.columns(2)
    with imp1:
        num_imp = st.selectbox(
            "Numeric imputation",
            ["Do nothing", "Mean", "Median", "Zero",
             "Group mean", "Group median", "KNN", "Iterative",
             "Multiple imputation"],
        )
    with imp2:
        cat_imp = st.selectbox(
            "Categorical imputation",
            ["Do nothing", "Mode", "Unknown", "Group mode"],
        )

    knn_k = 5
    iter_n = 10
    multi_n, multi_iter, multi_seed = 5, 10, 42
    if num_imp == "KNN":
        knn_k = st.slider("KNN neighbors", 2, 20, 5)
    if num_imp == "Iterative":
        iter_n = st.slider("Max iterations", 5, 50, 10, 5)
    if num_imp == "Multiple imputation":
        st.info("Multiple imputation creates several datasets. Download as ZIP.")
        mc1, mc2, mc3 = st.columns(3)
        multi_n    = mc1.slider("# datasets", 2, 20, 5)
        multi_iter = mc2.slider("Max iterations", 5, 50, 10, 5)
        multi_seed = int(mc3.number_input("Random seed", 1, value=42))

    st.markdown("---")
    st.markdown("### Outlier Handling")
    outlier_opt = st.selectbox(
        "Outlier method",
        ["Do nothing", "Cap using IQR", "Remove rows using IQR",
         "Cap using Z-Score", "Remove rows using Z-Score"],
    )
    z_clean = 3.0
    if "IQR" in outlier_opt:
        st.info("IQR: values outside Q1 − 1.5×IQR and Q3 + 1.5×IQR are treated as outliers.")
    if "Z-Score" in outlier_opt:
        z_clean = st.slider("Z-Score threshold", 1.5, 5.0, 3.0, 0.5, key="cl_z")
    if "Remove rows" in outlier_opt:
        st.warning("Row removal can delete many observations. Use with care.")

    # ── Preview impact of destructive operations ──────────
    is_destructive = (
        remove_dups
        or (drop_miss_thresh < 100)
        or "Remove rows" in outlier_opt
    )

    if is_destructive:
        preview_df = df.copy()
        rows_before = len(preview_df)

        if remove_dups:
            preview_df = preview_df.drop_duplicates()

        if drop_miss_thresh < 100:
            pct_missing = preview_df.isnull().mean() * 100
            cols_to_drop = pct_missing[pct_missing > drop_miss_thresh].index.tolist()
        else:
            cols_to_drop = []

        rows_after_dedup = len(preview_df)

        if "Remove rows" in outlier_opt and numeric_cols:
            preview_outlier_cols = [c for c in numeric_cols if c in preview_df.columns]
            if "Z-Score" in outlier_opt:
                for col in preview_outlier_cols:
                    s = preview_df[col].dropna()
                    if len(s) > 0 and s.std() > 0:
                        z = (preview_df[col] - s.mean()).abs() / s.std()
                        preview_df = preview_df[preview_df[col].isna() | (z <= z_clean)]
            else:
                for col in preview_outlier_cols:
                    s = preview_df[col].dropna()
                    if len(s) > 0:
                        q1, q3 = s.quantile(0.25), s.quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        preview_df = preview_df[
                            preview_df[col].isna() | preview_df[col].between(lower, upper)
                        ]

        rows_after_all = len(preview_df)
        rows_lost = rows_before - rows_after_all
        pct_lost = round(rows_lost / rows_before * 100, 1) if rows_before > 0 else 0

        st.markdown("### Preview Impact")
        pv1, pv2, pv3 = st.columns(3)
        pv1.metric("Rows before", f"{rows_before:,}")
        pv2.metric("Rows after", f"{rows_after_all:,}")
        pv3.metric(
            "Rows that will be removed",
            f"{rows_lost:,} ({pct_lost}%)",
        )

        if cols_to_drop:
            st.warning(
                "Columns that will be dropped (missing % above threshold): "
                + ", ".join(cols_to_drop)
            )

        if pct_lost > 30:
            st.error(
                "This will remove more than 30% of your rows ("
                + str(pct_lost) + "%). Review your settings before proceeding."
            )
        elif pct_lost > 10:
            st.warning(
                "This will remove " + str(pct_lost) + "% of your rows. "
                "Double-check this is intended."
            )
        elif rows_lost > 0:
            st.info(
                str(rows_lost) + " row(s) will be removed ("
                + str(pct_lost) + "%)."
            )

        confirm_destructive = st.checkbox(
            "I reviewed the impact above and want to proceed",
            key="cl_confirm_destructive",
        )
    else:
        confirm_destructive = True

    st.markdown("---")
    apply_disabled = is_destructive and not confirm_destructive
    if apply_disabled:
        st.caption("Check the confirmation box above to enable this button.")

    if st.button(
        "▶ Apply Cleaning", use_container_width=True,
        disabled=apply_disabled,
    ):
        with st.spinner("Applying cleaning and imputation…"):
            result = clean_dataset(
                df=df,
                remove_duplicates=remove_dups,
                remove_constant_cols=remove_const,
                remove_id_cols=remove_ids,
                drop_missing_threshold=drop_miss_thresh,
                numeric_imputation_method=(
                    "Do nothing" if num_imp == "Multiple imputation" else num_imp
                ),
                categorical_imputation_method=cat_imp,
                group_col=group_col,
                outlier_method=outlier_opt,
                possible_id_cols=possible_id_cols,
                constant_cols=constant_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                knn_neighbors=knn_k,
                iterative_max_iter=iter_n,
                z_score_threshold=z_clean,
            )
            st.session_state["df_cleaned"] = result
            st.session_state["cleaning_applied"] = True
            df_cleaned = result
        st.success("✅ Cleaning applied and saved for use in the Models tab.")

    if st.session_state["cleaning_applied"]:
        st.markdown("### After Cleaning")
        dc = st.session_state["df_cleaned"]
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Rows", f"{dc.shape[0]:,}", f"{dc.shape[0]-df.shape[0]:,}")
        a2.metric("Columns", f"{dc.shape[1]:,}", f"{dc.shape[1]-df.shape[1]:,}")
        a3.metric("Missing", f"{int(dc.isnull().sum().sum()):,}",
                  f"{int(dc.isnull().sum().sum())-int(df.isnull().sum().sum()):,}")
        a4.metric("Duplicates", f"{int(dc.duplicated().sum()):,}")
        st.dataframe(dc.head(20), use_container_width=True)

        st.download_button(
            "📥 Download cleaned dataset (CSV)",
            data=dc.to_csv(index=False).encode(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if num_imp == "Multiple imputation":
            st.markdown("### Download Multiple Imputed Datasets")
            with st.spinner(f"Creating {multi_n} imputed datasets…"):
                zip_buf = create_multiple_imputed_zip(
                    df=dc,
                    numeric_cols=dc.select_dtypes(include=np.number).columns.tolist(),
                    n_datasets=multi_n,
                    max_iter=multi_iter,
                    random_seed=multi_seed,
                )
            if zip_buf:
                st.download_button(
                    "📥 Download imputed datasets (ZIP)",
                    data=zip_buf,
                    file_name="multiple_imputed_datasets.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            else:
                st.warning("No usable numeric columns for multiple imputation.")

    # ── Advanced Row Filtering ─────────────────────────────
    st.markdown("---")
    st.markdown("### Advanced Row Filtering")
    st.caption(
        "Use this section to remove specific rows after cleaning — "
        "e.g. influential observations identified by Cook's Distance, "
        "impossible values, or rows matching a condition."
    )

    _dc = st.session_state["df_cleaned"] if st.session_state["df_cleaned"] is not None else df.copy()

    filter_method = st.radio(
        "Filter method",
        [
            "Remove rows by index",
            "Remove rows by condition",
            "Remove rows with missing values",
        ],
        horizontal=True,
        key="adv_filter_method",
    )

    if filter_method == "Remove rows by index":
        st.caption(
            "Enter the row index numbers to remove (comma-separated). "
            "You can find these in the Cook's Distance table under Models → Linear Regression → Diagnostics."
        )
        idx_input = st.text_input(
            "Row indices to remove",
            placeholder="e.g. 3813, 2790, 2787",
            key="adv_idx_input",
        )
        if idx_input.strip():
            try:
                indices_to_remove = [int(x.strip()) for x in idx_input.split(",") if x.strip()]
                valid = [i for i in indices_to_remove if i in _dc.index]
                invalid = [i for i in indices_to_remove if i not in _dc.index]
                st.info(
                    f"Will remove **{len(valid)}** row(s): {valid}"
                    + (f"  |  Not found: {invalid}" if invalid else "")
                )
                if valid and st.button("▶ Remove these rows", key="adv_idx_apply", use_container_width=True):
                    _dc = _dc.drop(index=valid, errors="ignore")
                    st.session_state["df_cleaned"] = _dc
                    st.session_state["cleaning_applied"] = True
                    st.success(f"✅ Removed {len(valid)} row(s). Dataset now has {len(_dc):,} rows.")
                    st.rerun()
            except ValueError:
                st.error("Invalid input — please enter numbers only, separated by commas.")

    elif filter_method == "Remove rows by condition":
        st.caption("Remove rows where a column satisfies a condition — e.g. impossible values or extreme outliers.")
        _num_cols_dc = _dc.select_dtypes(include=np.number).columns.tolist()
        _all_cols_dc = _dc.columns.tolist()

        cond_col1, cond_col2, cond_col3 = st.columns(3)
        cond_col = cond_col1.selectbox("Column", _all_cols_dc, key="adv_cond_col")
        cond_op = cond_col2.selectbox(
            "Condition",
            ["==", "!=", ">", ">=", "<", "<=", "contains", "is missing"],
            key="adv_cond_op",
        )
        cond_val = cond_col3.text_input("Value", placeholder="e.g. 100 or MD", key="adv_cond_val")

        if st.button("Preview rows to remove", key="adv_cond_preview", use_container_width=True):
            try:
                col_series = _dc[cond_col]
                if cond_op == "is missing":
                    mask = col_series.isna()
                elif cond_op == "contains":
                    mask = col_series.astype(str).str.contains(cond_val, na=False)
                else:
                    if cond_col in _num_cols_dc:
                        val = float(cond_val)
                        ops = {"==": col_series == val, "!=": col_series != val,
                               ">": col_series > val, ">=": col_series >= val,
                               "<": col_series < val, "<=": col_series <= val}
                    else:
                        ops = {"==": col_series.astype(str) == cond_val,
                               "!=": col_series.astype(str) != cond_val,
                               ">": col_series.astype(str) > cond_val,
                               ">=": col_series.astype(str) >= cond_val,
                               "<": col_series.astype(str) < cond_val,
                               "<=": col_series.astype(str) <= cond_val}
                    mask = ops[cond_op]
                n_match = int(mask.sum())
                st.info(f"**{n_match}** row(s) match this condition.")
                if n_match > 0:
                    st.dataframe(_dc[mask].head(20), use_container_width=True)
                    st.session_state["adv_cond_mask"] = mask
            except Exception as e:
                st.error("Could not apply condition: " + str(e))

        if st.session_state.get("adv_cond_mask") is not None:
            if st.button("▶ Remove matching rows", key="adv_cond_apply", use_container_width=True):
                mask = st.session_state["adv_cond_mask"]
                _dc = _dc[~mask].copy()
                st.session_state["df_cleaned"] = _dc
                st.session_state["cleaning_applied"] = True
                st.session_state["adv_cond_mask"] = None
                st.success(f"✅ Rows removed. Dataset now has {len(_dc):,} rows.")
                st.rerun()

    elif filter_method == "Remove rows with missing values":
        st.caption("Remove any row that has at least one missing value in the selected columns.")
        _all_cols_dc = _dc.columns.tolist()
        miss_cols = st.multiselect(
            "Apply to columns (leave empty = all columns)",
            _all_cols_dc,
            default=[],
            key="adv_miss_cols",
        )
        check_cols = miss_cols if miss_cols else _all_cols_dc
        n_missing_rows = int(_dc[check_cols].isnull().any(axis=1).sum())
        st.info(f"**{n_missing_rows}** row(s) have at least one missing value in the selected columns.")
        if n_missing_rows > 0 and st.button("▶ Remove rows with missing values", key="adv_miss_apply", use_container_width=True):
            _dc = _dc.dropna(subset=check_cols).copy()
            st.session_state["df_cleaned"] = _dc
            st.session_state["cleaning_applied"] = True
            st.success(f"✅ Removed {n_missing_rows} row(s). Dataset now has {len(_dc):,} rows.")
            st.rerun()

# ════════════════════════════════════════════════════
# Tab 4 — Visual Analysis

