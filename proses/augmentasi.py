import pandas as pd

# Fungsi untuk augmentasi data dengan penggeseran (shifting)
def augment_shift(df, y_col, shift_range):
    augmented_data = []
    for shift in range(-shift_range, shift_range + 1):
        df_shifted = df.copy()
        df_shifted[y_col] = df_shifted[y_col].shift(shift)
        df_shifted['Shift'] = shift
        augmented_data.append(df_shifted.dropna())
    
    augmented_df = pd.concat(augmented_data)
    augmented_df = augmented_df.reset_index(drop=True)
    
    return augmented_df