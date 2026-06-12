with open('modules/models/time_series.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Total lines:", len(lines))

for i, line in enumerate(lines):
    if 'coef_tbl = pd.DataFrame' in line and 'run_its' not in line:
        start = i
        print("Found at line:", i+1)
        for j in range(max(0,i-5), min(i+15, len(lines))):
            print(j+1, repr(lines[j]))
        break