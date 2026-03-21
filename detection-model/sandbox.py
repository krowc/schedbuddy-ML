    # Extract header names
    header_box = header_dets[0].bbox
    extracted = []
    for col in columns:
        header_cell = bbox_intersection(header_box, col.bbox)
        header_cell_text = ocr_crop(detector.image, header_cell) if header_cell else ""
        if "units" in header_cell_text.lower():
            header_cell_text = "Units"
        extracted.append(header_cell_text)

    if any(extracted):
        clean = [t or f"col_{i + 1}" for i, t in enumerate(extracted)]
        rows_as_dicts = [
            {clean[i]: row[header_names[i]] for i in range(n_cols)}
            for row in rows_as_dicts
        ]
        header_names = clean


    for r_idx, row in enumerate(rows, 1):
        row_dict: dict[str, str] = {}
        for c_idx, col in enumerate(columns, 1):
            box = bbox_intersection(row.bbox, col.bbox)
            text = ocr_crop(detector.image, box) if box else ""
            col_name = header_names[c_idx - 1]

            if col_name == "Units" :
                row_dict["Units"] = parse_units_cell(text)
            else:
                row_dict[col_name] = text
                
            cell_records.append(CellRecord(row=r_idx, column=c_idx, bbox=box, text=text))
        rows_as_dicts.append(row_dict)