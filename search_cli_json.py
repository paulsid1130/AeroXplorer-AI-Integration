import csv, json

def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def search_tags(data, keywords):
    results = []
    for row in data:
        text = " ".join(row.values()).lower()
        if all(k.lower() in text for k in keywords):
            results.append({
                "Filename": row["Filename"],
                "Airline_1": row["Airline_1"],
                "Airline_2": row["Airline_2"],
                "FlightState": row["FlightState"],
                "TimeOfDay": row["TimeOfDay"]
            })
    return results

def main():
    data = load_csv("mapped_tags_output.csv")
    query = input("Enter tags (comma separated): ")
    keywords = [k.strip() for k in query.split(",")]
    result = search_tags(data, keywords)
    print(json.dumps(result, indent=2) if result else "No matches found.")

if __name__ == "__main__":
    main()

