import csv


def get_name_mapping(csv_file):
    name_mapping = {}
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            old_name = row['old_name']
            name_mapping[name] = old_name
    return name_mapping


def get_descriptions():
    descriptions = {}
    with open('data/videos_descriptions.csv', mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['title']
            descriptions[name] = {
                'brief': row['brief'],
                'time': row['time'],
                'video_url': row['video_url'],
                'length': row['length'],
                # 'image': row['image'],
                # 'play_url': row['play_url'],
                # 'focus_date': row['focus_date'],
                # 'id': row['id'],
                # 'guid': row['guid'],
                # 'mode': row['mode'],
                'place': row['place'],
                'longitude': row['longitude'],
                'latitude': row['latitude'],
            }
    return descriptions


def merge_metadata(name_mapping, descriptions):
    metadata = {}
    for name, old_name in name_mapping.items():
        old_name = old_name.replace("-", "/").replace(".mp4", "")
        metadata[name] = descriptions.get(old_name)
        metadata[name]['title'] = old_name
    return metadata

def main():
    name_mapping = get_name_mapping('data/metadata/name_mapping.csv')
    descriptions = get_descriptions()
    metadata = merge_metadata(name_mapping, descriptions)
    with open('data/metadata/metadata.csv', 'w', encoding='utf-8', newline='') as f:
        field_names = list(metadata[next(iter(metadata))].keys())
        field_names.insert(0, 'name')
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for name, data in metadata.items():
            row = {'name': name}
            row.update(data)
            writer.writerow(row)


if __name__ == "__main__":
    main()