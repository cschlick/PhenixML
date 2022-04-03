# copied from the espaloma project. The intention is to replace in time.

import torch

def fp_rdkit(atom):
  from rdkit import Chem

  HYBRIDIZATION_RDKIT = {
      Chem.rdchem.HybridizationType.SP: torch.tensor(
          [1, 0, 0, 0, 0],
          dtype=torch.get_default_dtype(),
      ),
      Chem.rdchem.HybridizationType.SP2: torch.tensor(
          [0, 1, 0, 0, 0],
          dtype=torch.get_default_dtype(),
      ),
      Chem.rdchem.HybridizationType.SP3: torch.tensor(
          [0, 0, 1, 0, 0],
          dtype=torch.get_default_dtype(),
      ),
      Chem.rdchem.HybridizationType.SP3D: torch.tensor(
          [0, 0, 0, 1, 0],
          dtype=torch.get_default_dtype(),
      ),
      Chem.rdchem.HybridizationType.SP3D2: torch.tensor(
          [0, 0, 0, 0, 1],
          dtype=torch.get_default_dtype(),
      ),
      Chem.rdchem.HybridizationType.S: torch.tensor(
          [0, 0, 0, 0, 0],
          dtype=torch.get_default_dtype(),
      ),
  }
  return torch.cat(
      [
          torch.tensor(
              [
                  atom.GetTotalDegree(),
                  atom.GetTotalNumHs(),
                  atom.GetTotalValence(),
                  atom.GetExplicitValence(),
                  atom.GetFormalCharge(),
                  atom.GetIsAromatic() * 1.0,
                  atom.GetMass(),
                  atom.IsInRingSize(3) * 1.0,
                  atom.IsInRingSize(4) * 1.0,
                  atom.IsInRingSize(5) * 1.0,
                  atom.IsInRingSize(6) * 1.0,
                  atom.IsInRingSize(7) * 1.0,
                  atom.IsInRingSize(8) * 1.0,
              ],
              dtype=torch.get_default_dtype(),
          ),
          HYBRIDIZATION_RDKIT[atom.GetHybridization()],
      ],
      dim=0,
  )

  