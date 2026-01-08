let
  default = import ./default.nix;
  defaultPkgs = default.pkgs;
  defaultShell = default.shell;
  defaultBuildInputs = defaultShell.buildInputs;
  defaultConfigurePhase = ''
    cp ${./_rixpress/default_libraries.py} libraries.py
    cp ${./_rixpress/default_libraries.R} libraries.R
    mkdir -p $out  
    mkdir -p .julia_depot  
    export JULIA_DEPOT_PATH=$PWD/.julia_depot  
    export HOME_PATH=$PWD
  '';
  
  # Function to create Python derivations
  makePyDerivation = { name, buildInputs, configurePhase, buildPhase, src ? null }:
    let
      pickleFile = "${name}";
    in
      defaultPkgs.stdenv.mkDerivation {
        inherit name src;
        dontUnpack = true;
        buildInputs = buildInputs;
        inherit configurePhase buildPhase;
        installPhase = ''
          cp ${pickleFile} $out
        '';
      };

  # Define all derivations
    y_raw = makePyDerivation {
    name = "y_raw";
    src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./data/raw/y.xlsx ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src input_folder
python -c "
exec(open('libraries.py').read())
file_path = 'input_folder/data/raw/y.xlsx'
data = eval('lambda p: __import__(\'pandas\').read_excel(p, sheet_name=\'data_y\')')(file_path)
with open('y_raw', 'wb') as f:
    pickle.dump(data, f)
"
    '';
  };

  x_raw = makePyDerivation {
    name = "x_raw";
    src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./data/raw/x.xlsx ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src input_folder
python -c "
exec(open('libraries.py').read())
file_path = 'input_folder/data/raw/x.xlsx'
data = eval('lambda p: __import__(\'pandas\').read_excel(p, sheet_name=\'data_x\')')(file_path)
with open('x_raw', 'wb') as f:
    pickle.dump(data, f)
"
    '';
  };

  preds_json = makePyDerivation {
    name = "preds_json";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./pipeline/functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${y_raw}/y_raw', 'rb') as f: y_raw = pickle.load(f)
with open('${x_raw}/x_raw', 'rb') as f: x_raw = pickle.load(f)
exec(open('pipeline/functions.py').read())
exec('preds_json = run_hw2_pipeline(y_raw, x_raw)')
serialize_df_to_json(globals()['preds_json'], 'preds_json')
"
    '';
  };

  cells_zip = makePyDerivation {
    name = "cells_zip";
     src = defaultPkgs.lib.fileset.toSource {
      root = ./.;
      fileset = defaultPkgs.lib.fileset.unions [ ./pipeline/functions.py ];
    };
    buildInputs = defaultBuildInputs;
    configurePhase = defaultConfigurePhase;
    buildPhase = ''
      cp -r $src/* .
      python -c "
exec(open('libraries.py').read())
with open('${y_raw}/y_raw', 'rb') as f: y_raw = pickle.load(f)
with open('${x_raw}/x_raw', 'rb') as f: x_raw = pickle.load(f)
exec(open('pipeline/functions.py').read())
exec('cells_zip = build_cell_artifacts(y_raw, x_raw)')
zip_dir_to_file(globals()['cells_zip'], 'cells_zip')
"
    '';
  };

  # Generic default target that builds all derivations
  allDerivations = defaultPkgs.symlinkJoin {
    name = "all-derivations";
    paths = with builtins; attrValues { inherit y_raw x_raw preds_json cells_zip; };
  };

in
{
  inherit y_raw x_raw preds_json cells_zip;
  default = allDerivations;
}
