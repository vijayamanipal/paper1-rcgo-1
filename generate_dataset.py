"""
Generate a synthetic VulneraScope-X-style dataset (100 rows).

NOTE: This is a RECONSTRUCTION built from the schema in Table 1 of the paper.
It follows the described columns and spirit (synthetic CVE intelligence) so the
illustrative pipeline has data to run on.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 100

vendors = ["Canonical", "Google", "SAP", "Microsoft", "Oracle", "Adobe",
           "Apache", "Cisco", "RedHat", "Mozilla"]
products = {
    "Canonical": ["Ubuntu"], "Google": ["Chrome", "Android"],
    "SAP": ["SAP ERP", "NetWeaver"], "Microsoft": ["Windows", "Office"],
    "Oracle": ["MySQL", "WebLogic"], "Adobe": ["Acrobat", "Flash"],
    "Apache": ["HTTP Server", "Tomcat"], "Cisco": ["IOS", "AnyConnect"],
    "RedHat": ["Enterprise Linux"], "Mozilla": ["Firefox"],
}
attack_vectors = ["NETWORK", "ADJACENT", "LOCAL", "PHYSICAL"]
complexities = ["LOW", "HIGH"]
assigner_domains = ["mitre.org", "redhat.com", "google.com", "microsoft.com",
                    "oracle.com", "adobe.com", "cisco.com", "apache.org"]

rows = []
for i in range(N):
    vendor = rng.choice(vendors)
    product = rng.choice(products[vendor])
    year = int(rng.integers(2015, 2025))
    pub = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(int(rng.integers(0, 364)), unit="D")
    mod = pub + pd.to_timedelta(int(rng.integers(1, 400)), unit="D")
    cvss = round(float(rng.uniform(2.0, 10.0)), 1)
    hist_count = int(rng.integers(0, 120))
    exploit = int(rng.random() < (cvss / 12.0))          # higher CVSS -> more likely exploit
    patch = int(rng.random() < 0.65)
    # Label: "defective/high-risk" driven by severity, exploit presence, and history
    # weaker signal + substantial random noise so classes overlap realistically
    risk = 0.25 * (cvss / 10) + 0.20 * exploit + 0.10 * (hist_count / 120) + 0.45 * rng.random()
    label = int(risk > 0.5)
    if rng.random() < 0.15:          # 15% label noise
        label = 1 - label

    rows.append({
        "CVE_ID": f"CVE-{year}-{1000 + i}",
        "Assigner_Domain": rng.choice(assigner_domains),
        "Published_Date": pub.date().isoformat(),
        "Modified_Date": mod.date().isoformat(),
        "Description_Length": int(rng.integers(40, 600)),
        "Has_CWE": int(rng.random() < 0.8),
        "Reference_Count": int(rng.integers(1, 30)),
        "Product_Name": product,
        "Vendor_Name": vendor,
        "CVSS_Base_Score": cvss,
        "Attack_Vector": rng.choice(attack_vectors),
        "Attack_Complexity": rng.choice(complexities),
        "User_Interaction_Required": int(rng.random() < 0.4),
        "Historical_CVE_Count": hist_count,
        "Avg_Historical_CVSS": round(float(rng.uniform(3.0, 9.5)), 1),
        "Patch_Link_Available": patch,
        "Exploit_Exists": exploit,
        "Is_Defective": label,   # target variable
    })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/VulneraScope-X.csv", index=False)
print(df.head())
print("\nShape:", df.shape)
print("Class balance (Is_Defective):")
print(df["Is_Defective"].value_counts())
