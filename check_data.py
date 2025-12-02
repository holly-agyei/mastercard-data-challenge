import pandas as pd

# Check Excel files
print("=" * 60)
print("USA IGS FILE")
print("=" * 60)
df = pd.read_excel('usa_igs.xlsx', sheet_name='Compared to USA', header=None)
print('Shape:', df.shape)
print('First 10 rows:')
print(df.head(10).to_string())
print()
print("Column values in row 1:")
print(df.iloc[1].tolist())

print("\n" + "=" * 60)
print("STATE IGS FILE")
print("=" * 60)
df2 = pd.read_excel('igs_state.xlsx', sheet_name='Compared to State', header=None)
print('Shape:', df2.shape)
print('First 5 rows:')
print(df2.head(5).to_string())

print("\n" + "=" * 60)
print("RURAL IGS FILE")
print("=" * 60)
df3 = pd.read_excel('igs_rural.xlsx', sheet_name='Compared to Urban-Rural', header=None)
print('Shape:', df3.shape)
print('First 5 rows:')
print(df3.head(5).to_string())

