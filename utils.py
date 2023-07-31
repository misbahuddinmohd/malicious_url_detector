def makeTokens(f):
    tkns_BySlash = str(f).split('/')  # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')  # make tokens after splitting by dash
        for token in tokens:
            tkns_ByDot = token.split('.')  # make tokens after splitting by dot
            total_Tokens.append(token)  # add token to the list
            total_Tokens.extend(tkns_ByDot)  # add dot tokens to the list
    total_Tokens = list(OrderedDict.fromkeys(total_Tokens))  # remove redundant tokens while preserving order
    for itm in ['com']:
        if itm in total_Tokens:
            total_Tokens.remove(itm)
    return total_Tokens
