function sanity_check_gender_split()
% Sanity checks for male/female splits using speakerGenderMap.mat
% - No dependency on loadGenderSplitData subfunctions
% - Prints counts for train/test; audits mismatches; optional pitch spot-check

    % --- Load genderMap (from config if present, else local .mat) ---
    if exist('kws_config','file')
        cfg = kws_config();
        gmFile = cfg.paths.genderMapFile;
    else
        gmFile = 'speakerGenderMap.mat';
    end
    assert(exist(gmFile,'file')==2, 'Gender map file not found: %s', gmFile);
    S = load(gmFile,'genderMap');
    genderMap = S.genderMap;

    % --- Load dataset file lists (your existing helper) ---
    [trainingFiles, trainingLabels, testingFiles, testingLabels] = loadAudioData(); %#ok<ASGLU>

    % --- Build gender masks (standalone; no subfunction calls) ---
    keepF_train = genderMask(trainingFiles, genderMap, 'female');
    keepM_train = genderMask(trainingFiles, genderMap, 'male');
    keepF_test  = genderMask(testingFiles,  genderMap, 'female');
    keepM_test  = genderMask(testingFiles,  genderMap, 'male');

    unmappedTrain = ~(keepF_train | keepM_train);
    unmappedTest  = ~(keepF_test  | keepM_test );

    fprintf('Train female files: %6d | Train male files: %6d | Unmapped train: %6d\n', ...
            sum(keepF_train), sum(keepM_train), sum(unmappedTrain));
    fprintf('Test  female files: %6d | Test  male files: %6d | Unmapped test : %6d\n', ...
            sum(keepF_test),  sum(keepM_test),  sum(unmappedTest));

    % --- Audit: ensure mapped files actually match the requested gender ---
    A_F = auditGenderSplit(trainingFiles(keepF_train), genderMap, 'female');
    A_M = auditGenderSplit(trainingFiles(keepM_train), genderMap, 'male');

    fprintf('\nAUDIT (train): female mismatches=%d, unknown=%d | male mismatches=%d, unknown=%d\n', ...
            A_F.mismatch, A_F.unknown, A_M.mismatch, A_M.unknown);

    % --- Optional: spot-check with pitch-based estimator (if available) ---
    if exist('extract_gender','file')
        rng(0);
        nCheck = min(30, sum(keepF_train));
        if nCheck > 0
            idxF = find(keepF_train);
            samp = idxF(randperm(numel(idxF), nCheck));
            agree = 0;
            for ii = 1:numel(samp)
                ghat = extract_gender(trainingFiles{samp(ii)});
                if strcmpi(ghat,'female'), agree = agree + 1; end
            end
            fprintf('Pitch spot-check (female train): %d/%d agree (%.1f%%)\n', ...
                    agree, nCheck, 100*agree/max(1,nCheck));
        end
    else
        fprintf('(Note) extract_gender.m not found; skipping pitch spot-check.\n');
    end

    % --- Optional: write file lists for manual inspection ---
    outDir = 'GenderAudit';
    if ~exist(outDir,'dir'), mkdir(outDir); end
    writelines(string(trainingFiles(keepF_train)), fullfile(outDir,'train_female_files.txt'));
    writelines(string(trainingFiles(keepM_train)), fullfile(outDir,'train_male_files.txt'));
    writelines(string(testingFiles(keepF_test)),  fullfile(outDir,'test_female_files.txt'));
    writelines(string(testingFiles(keepM_test)),  fullfile(outDir,'test_male_files.txt'));
    fprintf('Wrote file lists to %s/ for manual audit.\n', outDir);
end

% --------- helpers (standalone) ----------
function mask = genderMask(fileList, genderMap, targetGender)
    n = numel(fileList);
    mask = false(n,1);
    for i = 1:n
        [~, base] = fileparts(fileList{i});  % e.g., 'speakerID_nohash_0'
        speakerID = strtok(base, '_');
        if isKey(genderMap, speakerID)
            mask(i) = strcmpi(genderMap(speakerID), targetGender);
        end
    end
end

function audit = auditGenderSplit(fileList, genderMap, targetGender)
    audit.total    = numel(fileList);
    audit.mapped   = 0;
    audit.mismatch = 0;
    audit.unknown  = 0;
    bad = {};
    for i = 1:numel(fileList)
        [~, base] = fileparts(fileList{i});
        speakerID = strtok(base, '_');
        if isKey(genderMap, speakerID)
            audit.mapped = audit.mapped + 1;
            if ~strcmpi(genderMap(speakerID), targetGender)
                audit.mismatch = audit.mismatch + 1;
                bad{end+1} = fileList{i}; %#ok<AGROW>
            end
        else
            audit.unknown = audit.unknown + 1;
            bad{end+1} = fileList{i}; %#ok<AGROW>
        end
    end
    audit.badList = bad(:);
end
