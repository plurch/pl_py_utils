
from typing import TypeVar

T = TypeVar('T')

def getIdxMappings(ids_list: list[T]) -> tuple[dict[T, int], dict[int, T]]:
  """
  Create bidirectional mappings between IDs and their indices in a list.

  Args:
      ids_list: A list of identifiers (integers or strings) to create mappings for. The list should contain unique ids.

  Returns:
      A tuple containing two dictionaries:
      - idToIdx: Maps each ID to its index position in the input list
      - idxToId: Maps each index position to its corresponding ID

  Example:
      >>> ids = ['apple', 'banana', 'cherry']
      >>> id_to_idx, idx_to_id = getIdxMappings(ids)
      >>> id_to_idx
      {'apple': 0, 'banana': 1, 'cherry': 2}
      >>> idx_to_id
      {0: 'apple', 1: 'banana', 2: 'cherry'}

  Note:
      The list should contain unique ids. If the input list contains duplicate IDs, later occurrences will
      overwrite earlier ones in the mapping dictionaries.
  """
  idToIdx = {}
  idxToId = {}

  for idx, id in enumerate(ids_list):
    idToIdx.update({id:idx})
    idxToId.update({idx:id})

  return idToIdx, idxToId
