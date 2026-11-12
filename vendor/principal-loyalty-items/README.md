# PrincipalBench settings

Service Contract, Car Sale and Bike Sale are adapted from three PrincipalBench items:

- Repository: <https://github.com/19PINE-AI/principal-loyalty> (MIT License, see `LICENSE`)
- Commit: `776e92137acc784f3088ab94587c6c13afed488c`
- Items: `pb-capit-anchored-01` (Service Contract), `pb-leak-price-01` (Car Sale),
  `pb-sanity-negot-01` (Bike Sale)

`stripped_down/v0/<id>.json` holds each party's private information (`user_1_private_info`,
`user_2_private_info`): the role and the private facts and preferences needed to play, taken from
the item's principal briefing and counterparty persona with additional instructions and strategy
directives removed. The public context, roles, proposal fields and payout rules are in
`configs/principal_bench/proposals/<id>.yaml`, which are not part of PrincipalBench.
