import re
from typing import Any, Optional, Union, List, Callable
from enum import Enum


class SortType(Enum):
    """Enumeration for different sort types"""

    ALPHABETICAL = "alphabetical"
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    AUTO = "auto"  # Auto-detect best sort type


class PathNavigator:
    """Utility class for navigating and sorting complex nested data structures using dot notation and array indexing"""

    @staticmethod
    def get_value_by_path(data: Any, path: str) -> tuple[bool, Any, str]:
        """
        Navigate to a value using a path like 'data.structure.socialLinks[0].platform'

        Args:
            data: The data structure to navigate
            path: Dot-notation path with optional array indexing

        Returns:
            tuple: (success, value, error_message)
        """
        try:
            current = data
            parts = PathNavigator._parse_path(path)

            for part in parts:
                if part["type"] == "key":
                    if isinstance(current, dict) and part["value"] in current:
                        current = current[part["value"]]
                    else:
                        return False, None, f"Key '{part['value']}' not found in dict"

                elif part["type"] == "index":
                    if isinstance(current, (list, tuple)) and 0 <= part["value"] < len(
                        current
                    ):
                        current = current[part["value"]]
                    else:
                        return (
                            False,
                            None,
                            f"Index {part['value']} out of range for list of length {len(current) if isinstance(current, (list, tuple)) else 'N/A'}",
                        )

            return True, current, ""

        except Exception as e:
            return False, None, f"Error navigating path '{path}': {str(e)}"

    @staticmethod
    def sort_array_by_path(
        data: Any,
        array_path: str,
        sort_key: Optional[str] = None,
        reverse: bool = False,
        sort_type: SortType = SortType.AUTO,
        limit: Optional[int] = None,
    ) -> tuple[bool, Any, str]:
        """
        Sort an array/list at the given path by a specified key or naturally.

        Args:
            data: The data structure to navigate
            array_path: Path to the array/list to sort
            sort_key: For arrays of objects, the property to sort by (supports dot notation)
            reverse: Whether to sort in descending order
            sort_type: Type of sorting to apply (auto-detected if AUTO)
            limit: Optional limit on number of items to return after sorting

        Returns:
            tuple: (success, sorted_array, error_message)
        """
        try:
            # Get the array at the specified path
            success, array_data, error = PathNavigator.get_value_by_path(
                data, array_path
            )
            if not success:
                return (
                    False,
                    None,
                    f"Could not find array at path '{array_path}': {error}",
                )

            if not isinstance(array_data, (list, tuple)):
                return (
                    False,
                    None,
                    f"Path '{array_path}' does not point to an array (found {type(array_data).__name__})",
                )

            if len(array_data) == 0:
                return True, [], "Array is empty, nothing to sort"

            # Convert to list for sorting
            array_list = list(array_data)

            try:
                if sort_key is None:
                    # Sort the array elements directly
                    sorted_array = PathNavigator._sort_simple_array(
                        array_list, reverse, sort_type
                    )
                else:
                    # Sort by a specific key within each array element
                    sorted_array = PathNavigator._sort_array_by_key(
                        array_list, sort_key, reverse, sort_type
                    )

                # Apply limit if specified
                if limit is not None and limit > 0:
                    sorted_array = sorted_array[:limit]

                return (
                    True,
                    sorted_array,
                    f"Successfully sorted {len(sorted_array)} items",
                )

            except Exception as sort_error:
                return False, None, f"Error during sorting: {str(sort_error)}"

        except Exception as e:
            return False, None, f"Error sorting array at path '{array_path}': {str(e)}"

    @staticmethod
    def _sort_simple_array(
        array_list: List[Any], reverse: bool, sort_type: SortType
    ) -> List[Any]:
        """Sort a simple array of values"""
        if sort_type == SortType.AUTO:
            sort_type = PathNavigator._detect_sort_type(array_list)

        if sort_type == SortType.NUMERICAL:
            # Try to convert to numbers, fallback to string sorting
            def safe_numeric_key(x):
                try:
                    if isinstance(x, (int, float)):
                        return x
                    elif isinstance(x, str):
                        # Try int first, then float
                        try:
                            return int(x)
                        except ValueError:
                            return float(x)
                    else:
                        return float(str(x))
                except (ValueError, TypeError):
                    return (
                        float("inf") if not reverse else float("-inf")
                    )  # Push invalid values to end

            return sorted(array_list, key=safe_numeric_key, reverse=reverse)

        elif sort_type == SortType.BOOLEAN:
            return sorted(array_list, key=lambda x: bool(x), reverse=reverse)

        else:  # ALPHABETICAL or fallback
            return sorted(array_list, key=lambda x: str(x).lower(), reverse=reverse)

    @staticmethod
    def _sort_array_by_key(
        array_list: List[Any], sort_key: str, reverse: bool, sort_type: SortType
    ) -> List[Any]:
        """Sort an array of objects by a specific key"""

        def get_sort_value(item):
            success, value, _ = PathNavigator.get_value_by_path(item, sort_key)
            return value if success else None

        # Get sample values to determine sort type if auto
        if sort_type == SortType.AUTO:
            sample_values = []
            for item in array_list[:10]:  # Sample first 10 items
                val = get_sort_value(item)
                if val is not None:
                    sample_values.append(val)
            sort_type = PathNavigator._detect_sort_type(sample_values)

        if sort_type == SortType.NUMERICAL:

            def numeric_sort_key(item):
                value = get_sort_value(item)
                try:
                    if isinstance(value, (int, float)):
                        return value
                    elif isinstance(value, str):
                        try:
                            return int(value)
                        except ValueError:
                            return float(value)
                    elif value is None:
                        return float("inf") if not reverse else float("-inf")
                    else:
                        return float(str(value))
                except (ValueError, TypeError):
                    return float("inf") if not reverse else float("-inf")

            return sorted(array_list, key=numeric_sort_key, reverse=reverse)

        elif sort_type == SortType.BOOLEAN:

            def bool_sort_key(item):
                value = get_sort_value(item)
                return bool(value) if value is not None else False

            return sorted(array_list, key=bool_sort_key, reverse=reverse)

        else:  # ALPHABETICAL or fallback

            def string_sort_key(item):
                value = get_sort_value(item)
                if value is None:
                    return "zzz_null" if not reverse else ""  # Push nulls to end
                return str(value).lower()

            return sorted(array_list, key=string_sort_key, reverse=reverse)

    @staticmethod
    def _detect_sort_type(values: List[Any]) -> SortType:
        """Auto-detect the best sort type for a list of values"""
        if not values:
            return SortType.ALPHABETICAL

        # Remove None values for type detection
        non_null_values = [v for v in values if v is not None]
        if not non_null_values:
            return SortType.ALPHABETICAL

        # Check if all values are boolean
        if all(isinstance(v, bool) for v in non_null_values):
            return SortType.BOOLEAN

        # Check if all values are numeric
        numeric_count = 0
        for value in non_null_values:
            if isinstance(value, (int, float)):
                numeric_count += 1
            elif isinstance(value, str):
                try:
                    float(value)
                    numeric_count += 1
                except ValueError:
                    pass

        # If most values are numeric, use numeric sorting
        if numeric_count / len(non_null_values) >= 0.8:
            return SortType.NUMERICAL

        return SortType.ALPHABETICAL

    @staticmethod
    def get_sortable_paths(
        data: Any, current_path: str = "", max_depth: int = 5
    ) -> List[dict]:
        """
        Get all paths that point to sortable arrays, with metadata about sort options.

        Args:
            data: Data structure to analyze
            current_path: Current path (for recursion)
            max_depth: Maximum depth to traverse

        Returns:
            List of dictionaries with path info and sort metadata
        """
        if max_depth <= 0:
            return []

        sortable_paths = []

        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{current_path}.{key}" if current_path else key

                    # Check if this value is a sortable array
                    if isinstance(value, (list, tuple)) and len(value) > 1:
                        sort_info = PathNavigator._analyze_array_for_sorting(
                            value, new_path
                        )
                        sortable_paths.append(sort_info)

                    # Recursively check nested structures
                    nested_sortable = PathNavigator.get_sortable_paths(
                        value, new_path, max_depth - 1
                    )
                    sortable_paths.extend(nested_sortable)

            elif isinstance(data, (list, tuple)) and len(data) > 1:
                for i, item in enumerate(data[:3]):  # Only check first 3 items
                    new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"

                    # Recursively check nested structures
                    nested_sortable = PathNavigator.get_sortable_paths(
                        item, new_path, max_depth - 1
                    )
                    sortable_paths.extend(nested_sortable)

        except Exception as e:
            pass  # Skip problematic structures

        return sortable_paths

    @staticmethod
    def _analyze_array_for_sorting(array_data: Union[list, tuple], path: str) -> dict:
        """Analyze an array to determine sorting capabilities"""
        info = {
            "path": path,
            "length": len(array_data),
            "can_sort_directly": True,
            "sort_keys": [],
            "detected_type": SortType.AUTO.value,
            "sample_values": [],
        }

        if not array_data:
            info["can_sort_directly"] = False
            return info

        # Sample first few items to understand structure
        sample_items = array_data[:5]
        info["sample_values"] = [
            str(item)[:50] + "..." if len(str(item)) > 50 else str(item)
            for item in sample_items
        ]

        # Detect if we can sort the array directly
        info["detected_type"] = PathNavigator._detect_sort_type(sample_items).value

        # If items are objects, find sortable keys
        if all(isinstance(item, dict) for item in sample_items):
            # Find common keys across all sample items
            all_keys = set()
            for item in sample_items:
                if isinstance(item, dict):
                    all_keys.update(item.keys())

            # Find keys that exist in most items and have sortable values
            sortable_keys = []
            for key in all_keys:
                values = []
                count = 0
                for item in sample_items:
                    if isinstance(item, dict) and key in item:
                        values.append(item[key])
                        count += 1

                # If key exists in most items and values are sortable
                if count >= len(sample_items) * 0.6:  # 60% threshold
                    sort_type = PathNavigator._detect_sort_type(values)
                    sortable_keys.append(
                        {
                            "key": key,
                            "type": sort_type.value,
                            "coverage": count / len(sample_items),
                            "sample_values": values[:3],
                        }
                    )

            info["sort_keys"] = sorted(
                sortable_keys, key=lambda x: x["coverage"], reverse=True
            )

        return info

    @staticmethod
    def _parse_path(path: str) -> list[dict]:
        """
        Parse a path like 'data.structure.socialLinks[0].platform' into components

        Returns:
            List of path components with type and value
        """
        parts = []

        # Split on dots but handle array notation
        segments = path.split(".")

        for segment in segments:
            # Check if this segment has array notation
            array_matches = re.findall(r"([^[]+)(\[[0-9]+\])", segment)

            if array_matches:
                # Handle key with array index like 'socialLinks[0]'
                key_part, index_part = array_matches[0]
                parts.append({"type": "key", "value": key_part})

                # Extract the index number
                index = int(index_part[1:-1])  # Remove [ and ]
                parts.append({"type": "index", "value": index})

            elif segment.startswith("[") and segment.endswith("]"):
                # Pure array index like '[0]'
                index = int(segment[1:-1])
                parts.append({"type": "index", "value": index})

            else:
                # Regular key
                if segment:  # Skip empty segments
                    parts.append({"type": "key", "value": segment})

        return parts

    @staticmethod
    def get_all_available_paths(
        data: Any, current_path: str = "", max_depth: int = 5
    ) -> list[str]:
        """
        Get all available paths in a data structure

        Args:
            data: Data structure to analyze
            current_path: Current path (for recursion)
            max_depth: Maximum depth to traverse

        Returns:
            List of all valid paths
        """
        if max_depth <= 0:
            return []

        paths = []

        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    paths.append(new_path)

                    # Recursively get paths from nested structures
                    nested_paths = PathNavigator.get_all_available_paths(
                        value, new_path, max_depth - 1
                    )
                    paths.extend(nested_paths)

            elif isinstance(data, (list, tuple)):
                for i, item in enumerate(data[:3]):  # Only check first 3 items
                    new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                    paths.append(new_path)

                    # Recursively get paths from nested structures
                    nested_paths = PathNavigator.get_all_available_paths(
                        item, new_path, max_depth - 1
                    )
                    paths.extend(nested_paths)

        except Exception as e:
            pass  # Skip problematic structures

        return paths
