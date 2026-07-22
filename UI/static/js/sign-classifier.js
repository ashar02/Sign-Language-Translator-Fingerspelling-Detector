/**
 * Browser RandomForest classifier for ASL fingerspelling.
 * Model file: /static/model/sign_rf.json (exported from model.p)
 */

const SignClassifier = (() => {
    let model = null;

    async function load(url = '/static/model/sign_rf.json') {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load classifier model (${response.status})`);
        }
        model = await response.json();
        return model;
    }

    function isReady() {
        return !!model;
    }

    /** Same 42-D normalization as UI/app.py process_hand_landmarks */
    function featuresFromLandmarks(landmarks) {
        const xs = landmarks.map(p => p.x);
        const ys = landmarks.map(p => p.y);
        const minX = Math.min(...xs);
        const minY = Math.min(...ys);
        const features = [];
        for (const p of landmarks) {
            features.push(p.x - minX);
            features.push(p.y - minY);
        }
        return features;
    }

    function predictTree(tree, features) {
        let node = 0;
        while (tree.l[node] !== -1) {
            const feat = tree.f[node];
            node = features[feat] <= tree.t[node] ? tree.l[node] : tree.r[node];
        }
        return tree.c[node];
    }

    /**
     * @returns {{ label: string, classIndex: number, votes: number }}
     */
    function predict(features) {
        if (!model) {
            throw new Error('Classifier not loaded');
        }
        if (!features || features.length !== model.n_features) {
            return { label: '', classIndex: -1, votes: 0 };
        }

        const voteCounts = new Map();
        for (const tree of model.trees) {
            const classPos = predictTree(tree, features);
            const classId = model.classes[classPos];
            voteCounts.set(classId, (voteCounts.get(classId) || 0) + 1);
        }

        let bestClass = model.classes[0];
        let bestVotes = -1;
        for (const [classId, votes] of voteCounts.entries()) {
            if (votes > bestVotes) {
                bestVotes = votes;
                bestClass = classId;
            }
        }

        // Match server labels_dict: 0->a ... 25->z, displayed uppercase in UI
        const label = String.fromCharCode(97 + Number(bestClass)).toUpperCase();
        return { label, classIndex: Number(bestClass), votes: bestVotes };
    }

    function predictFromLandmarks(landmarks) {
        return predict(featuresFromLandmarks(landmarks));
    }

    return { load, isReady, featuresFromLandmarks, predict, predictFromLandmarks };
})();

window.SignClassifier = SignClassifier;
