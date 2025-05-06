import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Prepare balanced regression dataset from Food.com reviews"
    )
    parser.add_argument('--recipes',      required=True, help='Path to RAW_recipes.csv')
    parser.add_argument('--interactions', required=True, help='Path to RAW_interactions.csv')
    parser.add_argument('--min_reviews',  type=int, default=50, help='Minimum reviews required per user')
    parser.add_argument('--seed',         type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--output_csv',   default='balanced_reviews.csv', help='Output CSV filename')
    args = parser.parse_args()

    # Load
    recipes     = pd.read_csv(args.recipes)
    interactions = pd.read_csv(args.interactions)

    # Build menu_text
    recipes['menu_text'] = recipes['Name'] + ' | ' + recipes['Description'].fillna('')
    df = interactions.merge(
        recipes[['RecipeId','menu_text']],
        on='RecipeId', how='inner'
    )

    # Rename
    df = df.rename(columns={
        'AuthorId':      'user_id',
        'Rating':        'label',
        'Review':        'review_text',
        'DateSubmitted': 'date'
    })

    # Parse date and sort
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values(['user_id','date']).reset_index(drop=True)

    # Filter users
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= args.min_reviews].index
    df = df[df['user_id'].isin(valid_users)].reset_index(drop=True)

    # Define rating bins [0-1),[1-2),[2-3),[3-4),[4-5]
    bins = [0,1,2,3,4,5]
    df['rating_bin'] = pd.cut(
        df['label'], bins=bins, include_lowest=True, right=False,
        labels=['0-1','1-2','2-3','3-4','4-5']
    )

    # Determine smallest bin size
    counts = df['rating_bin'].value_counts()
    min_count = counts.min()
    print("Bin counts before balancing:\n", counts.to_dict())
    print(f"Sampling {min_count} examples per bin to balance dataset.")

    # Sample equal number from each bin
    balanced_dfs = []
    for bin_label, group in df.groupby('rating_bin'):
        sampled = group.sample(n=min_count, random_state=args.seed)
        balanced_dfs.append(sampled)
    balanced = pd.concat(balanced_dfs).reset_index(drop=True)

    # Drop helper column and save
    balanced = balanced.drop(columns=['rating_bin'])
    balanced.to_csv(args.output_csv, index=False)
    print(f"Saved {len(balanced)} balanced reviews to {args.output_csv}")

if __name__ == '__main__':
    main()
