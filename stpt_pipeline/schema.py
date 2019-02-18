import voluptuous as vo

schema = vo.Schema({vo.Required('datalen'): vo.Range(1, 1000)})
