_pad = "_"
#_punctuation = ';:,.!|?¡¿—…"«»“” '
#_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
#_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ‖*"

#_data_letters_ipa = "̃"
#_data_letters  = "àâæçéèêëîïôœùûüÿŸÜÛÙŒÔÏÎËÊÈÉÇÆÂÀ"

letters = "*./|‖̃0123456789 aɑbcdDeəɛfɡijklmMnɲŋoøœɔpqrʁsʃtuUɥvwyYzʒ"
# # Export all symbols:
#symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_data_letters_ipa) + list(_data_letters)

symbols = [_pad] + list(letters)

# # Special symbol ids
SPACE_ID = symbols.index(" ")
