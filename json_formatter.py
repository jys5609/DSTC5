import json

from collections import defaultdict
from scripts.dataset_walker import dataset_walker

class JSONFormatter(object):

    def __init__(self, dataset_name, wall_time, suids, slot_value_dicts):
        """Format results to match Main task json format requirements

        :param dataset_name: the name of the dataset over which the tracker has been run (string)
        :param wall_time: the time in seconds it took to runt the tracker (float)
        :param suids: the list composed of a session-utterance id for each utterance (list of str)
        :param slot_value_dicts: the list of slot-value dict for each utterance (list of dictionary)
        """
        self.dataset_name = dataset_name
        self.wall_time = wall_time
        self.suids = suids
        self.slot_value_dicts = slot_value_dicts

        ### process into dictionaty
        sessions = []
        last_session_id = None
        for idx, suid in enumerate(suids):
            # extract session & utterance id
            assert len(suid) == 12
            session_id = int(suid[1:6])
            utter_id = int(suid[7:])
            # if session_id changes, dump
            if session_id != last_session_id:
                # dump except first iteration
                if last_session_id != None:
                    session['session_id'] = last_session_id
                    session['utterances'] = utterances
                    sessions.append(session)
                # reset session and utterances
                session = dict()
                utterances = []
            # append prediction result
            utterance = dict()
            utterance['utter_index'] = utter_id
            slot_value_dict = slot_value_dicts[idx]
            utterance['frame_label'] = slot_value_dict
            utterances.append(utterance)
            # prepare for next iteration
            last_session_id = session_id
        # add last session to data
        session['session_id'] = last_session_id
        session['utterances'] = utterances
        sessions.append(session)
        # finalize dict
        self.data = dict()
        self.data['dataset'] = self.dataset_name
        self.data['wall_time'] = wall_time
        self.data['sessions'] = sessions

        # iterate dataset and fill in the holes
        sessions = dataset_walker(self.dataset_name, dataroot='data', labels=False)
        for session, my_session in zip(sessions, self.data["sessions"]):
            session_id = session.log["session_id"]
            assert session_id == my_session["session_id"]

            my_utters = my_session["utterances"]
            for idx, (log_utter, _, _) in enumerate(session):
                if len(my_utters) <= idx or \
                    log_utter["utter_index"] != my_utters[idx]["utter_index"]:
                    my_utters.insert(idx, {
                        "frame_label": {},
                        "utter_index": log_utter["utter_index"]
                    })
            assert len(my_utters) == len(session)

        # preconvert into text
        self.json = json.dumps(self.data)
        self.pretty_json = json.dumps(self.data, sort_keys=True, indent=4)

    def __repr__(self):
        return self.pretty_json

    def __str__(self):
        return self.pretty_json

    def dump_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.pretty_json)
