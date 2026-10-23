# Verification pack

Everything below comes from stored files (raw CSV, `outputs/interim/*.parquet`, `outputs/*.json`); nothing was re-trained. Sections 1–4 are inputs for manual checking; the values computed by the code are only in section 5.

User: `11310850331381393395` — chosen with seed 42 from the 1150 cohort-A users, excluding the two users of `two_users.md`. Airing: chosen with seed 42 among this user's weeks 21–30 airings that involve ≥ 2 sessions.

## 1. Rules for the manual computation

* Airing end = schedule `start_time` (`DD.MM.YYYY HH:MM:SS`) + `duration` seconds.
* Use the log's `start_time` / `stop_time` (`YYYY-MM-DD HH:MM:SS`), not `duraton`; no time shift.
* Merge this user's sessions on this channel that overlap each other (sessions that only touch stay separate), intersect the result with the airing, sum the intersections, clip to `duration`.
* screen_time = overlap / duration; the airing counts as **watched** when screen_time ≥ 0.8.
* Truth = the user's top-5 programmes by number of watched airings in ISO weeks 21–30; ties → larger summed overlap of the watched airings, then smaller numeric id.

## 2. Raw schedule rows, channel 1519 (± a few hours; the airing to check is in **bold**)

| file | channel_id | channel_title | start_time | duration | tv_show_title | tv_show_id | tv_show_category |
|---|---|---|---|---|---|---|---|
| 21-30 | 1519 | ViP Comedy CEE | 21.05.2020 22:40:00 | 5700 | Х/ф 'Американский пирог'. | 90022919 | Фильмы |
| **21-30** | **1519** | **ViP Comedy CEE** | **22.05.2020 00:15:00** | **6300** | **Х/ф 'Американский пирог 2'.** | **90052498** | **Фильмы** |
| 21-30 | 1519 | ViP Comedy CEE | 22.05.2020 02:00:00 | 7800 | Х/ф 'Высокая мода'. | 90037017 | Фильмы |

## 3. Raw log rows: user `11310850331381393395`, `vsetv_id` = 1519, sessions within ±1 h of the airing

| user_id | vsetv_id | start_time | stop_time | duraton |
|---|---|---|---|---|
| 11310850331381393395 | 1519 | 2020-05-22 01:01:13 | 2020-05-22 01:07:39 | 386 |
| 11310850331381393395 | 1519 | 2020-05-22 01:16:37 | 2020-05-22 01:49:06 | 1949 |
| 11310850331381393395 | 1519 | 2020-05-22 01:53:03 | 2020-05-22 01:55:05 | 122 |

## 4. Inputs for the truth and the v7 recommendation

**This user's programmes in weeks 21–30** (from `airing_views.parquet`: `pairs` = (user, airing) pairs with any overlap, `watched_airings` = pairs with screen_time ≥ 0.8, `overlap_watched_s` = summed overlap of those watched airings, for the tie rule). Sorted by the truth rule.

| tv_show_id | Title | Category | pairs | watched_airings | overlap_watched_s |
|---|---|---|---|---|---|
| 10001596 | Т/с 'Жизнь как приговор', 2 сезон, 5 с. 'Раздавленный'. | Сериалы | 3 | 2 | 5400 |
| 200405 | 'Мамахохотала-шоу'. Лучшее. | Развлечения | 2 | 1 | 3900 |
| 1100411 | 'Вещдок'. | Познавательное | 2 | 1 | 3482 |
| 2400467 | 'Викна-Новости'. | Инфо | 4 | 1 | 3000 |
| 2400480 | ТСН. | Инфо | 3 | 1 | 2700 |
| 240081 | Факты. | Инфо | 2 | 1 | 1500 |
| 90043769 | Х/ф 'Иван Васильевич меняет профессию'. | Фильмы | 2 | 1 | 600 |
| 200352 | 'Вечерний квартал'. | Развлечения | 2 | 0 | 0 |
| 200393 | 'Бойцовский клуб'. | Развлечения | 1 | 0 | 0 |
| 500331 | 'Один за всех'. | Развлечения | 1 | 0 | 0 |
| 1000723 | Т/с 'Охотники за древностями', 3 сезон, 22 с. | Сериалы | 1 | 0 | 0 |
| 2400508 | Сегодня. | Инфо | 2 | 0 | 0 |
| 2500430 | 'Затерянный мир'. | Познавательное | 1 | 0 | 0 |
| 5200303 | 'Следствие вели...' с Леонидом Каневским. | Познавательное | 1 | 0 | 0 |
| 9009664 | Х/ф 'Грязные танцы'. | Фильмы | 1 | 0 | 0 |
| 10001261 | Т/с 'Мыслить как преступник'. | Сериалы | 1 | 0 | 0 |
| 10001499 | Т/с 'Вербное воскресенье', 1 с. | Сериалы | 1 | 0 | 0 |
| 10001817 | Т/с 'Ясновидец', 7 сезон, 16 с. | Сериалы | 1 | 0 | 0 |
| 12002753 | М/с 'Санджей и Крейг'. | Детям | 1 | 0 | 0 |
| 90013755 | Х/ф 'Игрок'. | Фильмы | 1 | 0 | 0 |
| 90016397 | Х/ф 'Сладкий и гадкий' | Фильмы | 1 | 0 | 0 |
| 90016788 | Х/ф 'Миротворец'. | Фильмы | 1 | 0 | 0 |
| 90023665 | Х/ф 'Робин Гуд: Принц воров'. | Фильмы | 1 | 0 | 0 |
| 90026635 | Х/ф 'Брак по-итальянски'. | Фильмы | 1 | 0 | 0 |
| 90029840 | Х/ф 'Мачеха'. | Фильмы | 1 | 0 | 0 |
| 90031615 | Х/ф 'Побег из Шоушенка'. | Фильмы | 1 | 0 | 0 |
| 90033419 | Х/ф 'Дракула'. | Фильмы | 1 | 0 | 0 |
| 90035553 | Х/ф 'Водный мир'. | Фильмы | 1 | 0 | 0 |
| 90036372 | Х/ф 'Операция 'Ы' и другие приключения Шурика'. | Фильмы | 1 | 0 | 0 |
| 90041727 | Х/ф 'Вспомнить все'. | Фильмы | 1 | 0 | 0 |
| 90045031 | Х/ф 'Дневной свет'. | Фильмы | 1 | 0 | 0 |
| 90045696 | Х/ф 'Пуленепробиваемый'. | Фильмы | 1 | 0 | 0 |
| 90048003 | Х/ф 'Пока ты спал'. | Фильмы | 1 | 0 | 0 |
| 90049405 | Х/ф 'Западня'. | Фильмы | 1 | 0 | 0 |
| 90049521 | Х/ф 'Очень страшное кино 2'. | Фильмы | 1 | 0 | 0 |
| 90052075 | Х/ф 'Три икса'. | Фильмы | 1 | 0 | 0 |
| 90052498 | Х/ф 'Американский пирог 2'. | Фильмы | 3 | 0 | 0 |
| 90052591 | Х/ф 'Миллионер поневоле'. | Фильмы | 1 | 0 | 0 |
| 90054535 | Х/ф 'Другой мир'. | Фильмы | 1 | 0 | 0 |
| 90054713 | Х/ф 'Фрида'. | Фильмы | 1 | 0 | 0 |
| 90056238 | Х/ф 'Тайное окно'. | Фильмы | 1 | 0 | 0 |
| 90056319 | Х/ф 'Гарри Поттер и тайная комната'. | Фильмы | 1 | 0 | 0 |
| 90056569 | Х/ф 'Ван Хельсинг'. | Фильмы | 1 | 0 | 0 |
| 90056711 | Х/ф 'Идальго: Погоня в пустыне'. | Фильмы | 1 | 0 | 0 |
| 90057266 | Х/ф 'Терминатор 3: Восстание машин'. | Фильмы | 1 | 0 | 0 |
| 90057449 | Х/ф 'Убить Билла 2'. | Фильмы | 2 | 0 | 0 |
| 90058853 | Х/ф 'После заката' | Фильмы | 1 | 0 | 0 |
| 90058930 | Х/ф 'Американский пирог 3: Свадьба'. | Фильмы | 1 | 0 | 0 |
| 90059503 | Х/ф 'Три икса 2: Новый уровень'. | Фильмы | 1 | 0 | 0 |
| 90059901 | Х/ф 'Изнанка судьбы'. | Фильмы | 1 | 0 | 0 |
| 90062359 | Х/ф 'Гарри Поттер и Кубок огня'. | Фильмы | 1 | 0 | 0 |
| 90063178 | Х/ф 'Аферисты Дик и Джейн'. | Фильмы | 1 | 0 | 0 |
| 90064231 | Х/ф 'Капоте'. | Фильмы | 1 | 0 | 0 |
| 90065042 | Х/ф 'Будь круче'. | Фильмы | 1 | 0 | 0 |
| 90065444 | Х/ф 'Развод по-американски'. | Фильмы | 1 | 0 | 0 |
| 90065473 | Х/ф 'Эван Всемогущий'. | Фильмы | 1 | 0 | 0 |
| 90066068 | Х/ф 'Мюнхен'. | Фильмы | 1 | 0 | 0 |
| 90066252 | Х/ф 'Идеальный незнакомец'. | Фильмы | 1 | 0 | 0 |
| 90067011 | Х/ф 'Через Вселенную'. | Фильмы | 1 | 0 | 0 |
| 90067333 | Х/ф 'Грейси'. | Фильмы | 1 | 0 | 0 |
| 90067845 | Х/ф 'Два дня в Париже'. | Фильмы | 1 | 0 | 0 |
| 90068005 | Х/ф 'Перевозчик 3'. | Фильмы | 1 | 0 | 0 |
| 90068169 | Х/ф 'Хэнкок'. | Фильмы | 1 | 0 | 0 |
| 90068760 | Х/ф 'Мальчикам это нравится'. | Фильмы | 1 | 0 | 0 |
| 90069354 | Х/ф 'Операция 'Валькирия'. | Фильмы | 1 | 0 | 0 |
| 90070062 | Х/ф 'Посылка'. | Фильмы | 1 | 0 | 0 |
| 90071020 | Х/ф 'Район №9'. | Фильмы | 1 | 0 | 0 |
| 90071327 | Х/ф 'Пипец'. | Фильмы | 1 | 0 | 0 |
| 90072464 | Х/ф 'Днепровский рубеж'. | Фильмы | 1 | 0 | 0 |
| 90072715 | Х/ф 'Области тьмы'. | Фильмы | 2 | 0 | 0 |
| 90073323 | Х/ф 'Буря'. | Фильмы | 1 | 0 | 0 |
| 90073473 | Х/ф 'Очень плохая училка'. | Фильмы | 1 | 0 | 0 |
| 90074592 | Х/ф 'Миссия невыполнима: Протокол Фантом'. | Фильмы | 1 | 0 | 0 |
| 90074629 | Х/ф 'Американский пирог: Все в сборе'. | Фильмы | 1 | 0 | 0 |
| 90074863 | Х/ф 'Папа-досвидос'. | Фильмы | 1 | 0 | 0 |
| 90074884 | Х/ф 'Параллельные миры'. | Фильмы | 1 | 0 | 0 |
| 90075111 | Х/ф 'Облачный атлас'. | Фильмы | 1 | 0 | 0 |
| 90076295 | Х/ф 'Призрачный патруль'. | Фильмы | 1 | 0 | 0 |
| 90076683 | Х/ф 'Колония'. | Фильмы | 1 | 0 | 0 |
| 90076885 | Х/ф 'Уцелевший'. | Фильмы | 1 | 0 | 0 |
| 90077028 | Х/ф 'Сезон убийц'. | Фильмы | 1 | 0 | 0 |
| 90077396 | Х/ф 'А вот и она'. | Фильмы | 1 | 0 | 0 |
| 90077857 | Х/ф 'Эверли'. | Фильмы | 1 | 0 | 0 |
| 90077862 | Х/ф 'Гонка'. | Фильмы | 1 | 0 | 0 |
| 90078086 | Х/ф 'Геркулес'. | Фильмы | 1 | 0 | 0 |
| 90078815 | Х/ф 'Прогулка'. | Фильмы | 1 | 0 | 0 |
| 90079260 | Х/ф 'Работорговля'. | Фильмы | 1 | 0 | 0 |
| 90079286 | Х/ф 'Выживший'. | Фильмы | 1 | 0 | 0 |
| 90079620 | Х/ф 'Гордость и предубеждение и зомби'. | Фильмы | 1 | 0 | 0 |
| 90079741 | Х/ф 'Крид: Наследие Рокки'. | Фильмы | 1 | 0 | 0 |
| 90079810 | Х/ф 'Великолепная семёрка'. | Фильмы | 1 | 0 | 0 |
| 90080482 | Х/ф 'Стражи Галактики. Часть 2'. | Фильмы | 1 | 0 | 0 |
| 90080655 | Х/ф 'Сторожевая застава', 1-4 с. | Фильмы | 1 | 0 | 0 |
| 90080730 | Х/ф 'Борг/Макинрой'. | Фильмы | 1 | 0 | 0 |
| 90080905 | Х/ф '24 часа на жизнь'. | Фильмы | 1 | 0 | 0 |
| 90081074 | Х/ф 'Винчестер. Дом, который построили призраки'. | Фильмы | 1 | 0 | 0 |
| 90081397 | Х/ф 'Талли'. | Фильмы | 1 | 0 | 0 |
| 90081539 | Х/ф 'Будет страшно - улыбайся', 1 ч. | Фильмы | 1 | 0 | 0 |
| 90081566 | Х/ф 'Кин'. | Фильмы | 1 | 0 | 0 |
| 90082007 | Х/ф 'Сильнее'. | Фильмы | 1 | 0 | 0 |
| 90082049 | Х/ф 'Алита: Боевой ангел'. | Фильмы | 1 | 0 | 0 |
| 90082063 | Х/ф 'Неслучайные встречи', 1-4 с. | Фильмы | 1 | 0 | 0 |
| 90082199 | Х/ф 'Любовь с закрытыми глазами', 1 и 2 с. | Фильмы | 1 | 0 | 0 |
| 90082209 | Х/ф 'Поезд на Париж'. | Фильмы | 1 | 0 | 0 |
| 90082445 | Х/ф 'Добро пожаловать домой'. | Фильмы | 1 | 0 | 0 |
| 90082687 | Х/ф 'Карательница'. | Фильмы | 1 | 0 | 0 |
| 90082767 | Х/ф 'Страшные истории для рассказа в темноте'. | Фильмы | 1 | 0 | 0 |
| 90082798 | Х/ф 'Экспедиция Юрского периода' | Фильмы | 1 | 0 | 0 |
| 120079318 | М/ф 'Холодное сердце'. | Детям | 1 | 0 | 0 |

**v7 top-5 for this user** (validation, out-of-fold; `outputs/interim/preds_val_v7.parquet`)

| rank | tv_show_id | Title | Category |
|---|---|---|---|
| 1 | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям |
| 2 | 240081 | Факты. | Инфо |
| 3 | 2400480 | ТСН. | Инфо |
| 4 | 1000589 | Т/с 'Отчаянные домохозяйки', 1 сезон, 10 с. 'Возвращайся'. | Сериалы |
| 5 | 10001261 | Т/с 'Мыслить как преступник'. | Сериалы |

## 5. Values computed by the code (check your results against these)

* Airing: channel 1519, tv_show_id 90052498, start 2020-05-22 00:15:00, duration 6300 s.
* `overlap_s` = **2457**, `n_sessions` (merged sessions touching the airing) = 3, `screen_time` = **0.390000**, `watched` = **False**.
* Truth (weeks 21–30, top-5): **10001596 200405 1100411 2400467 2400480**.
* v7 top-5: 12001682 240081 2400480 1000589 10001261; hits: 1; **AP@5 = 0.0667**.

## 6. Every number in README.md → its json key

| README number | Value in README | File | Key path | Value in json | in README text |
|---|---|---|---|---|---|
| Users to predict | 1,260 | `data_audit.json` | `submission.distinct_users` | 1260 | yes |
| Cohort A size | 1,150 | `results_validation.json` | `protocol.data_accounting.cohort_sizes.A` | 1150 | yes |
| Cohort C size | 4,030 | `results_validation.json` | `protocol.data_accounting.cohort_sizes.C` | 4030 | yes |
| Final file md5 | d0e51925c3911a9174817c3de129e94c | `results_final.json` | `md5` | d0e51925c3911a9174817c3de129e94c | yes |
| Kaggle private v7 (graded) | 0.43671 | `results_final.json` | `kaggle_submissions.best_private.kaggle_private` | 0.43671 | yes |
| Kaggle public v7 (reference) | 0.43802 | `results_final.json` | `kaggle_submissions.best_private.kaggle_public_reference` | 0.43802 | yes |
| Popularity MAP@5 A | 0.2357 | `results_validation.json` | `methods.popularity.map@5_A` | 0.235699 | yes |
| Popularity MAP@5 C | 0.2138 | `results_validation.json` | `methods.popularity.map@5_C` | 0.213795 | yes |
| Personal history MAP@5 A | 0.3438 | `results_validation.json` | `methods.personal_history.map@5_A` | 0.343832 | yes |
| Personal history MAP@5 C | 0.3042 | `results_validation.json` | `methods.personal_history.map@5_C` | 0.30421 | yes |
| MF (ALS) MAP@5 A | 0.1675 | `results_validation.json` | `methods.mf.map@5_A` | 0.167461 | yes |
| MF (ALS) MAP@5 C | 0.1615 | `results_validation.json` | `methods.mf.map@5_C` | 0.161546 | yes |
| CBF MAP@5 A | 0.2071 | `results_validation.json` | `methods.cbf.map@5_A` | 0.207052 | yes |
| CBF MAP@5 C | 0.1824 | `results_validation.json` | `methods.cbf.map@5_C` | 0.182445 | yes |
| v1 PPR MAP@5 A | 0.3549 | `results_validation.json` | `methods.ppr.map@5_A` | 0.354913 | yes |
| v1 PPR MAP@5 C | 0.3180 | `results_validation.json` | `methods.ppr.map@5_C` | 0.317993 | yes |
| v2 MAP@5 A | 0.3667 | `results_validation.json` | `improve.ppr.best.map@5_A` | 0.366742 | yes |
| v2 MAP@5 C | 0.3294 | `results_validation.json` | `improve.ppr.best.map@5_C` | 0.329417 | yes |
| v5 MAP@5 A | 0.3776 | `results_validation.json` | `improve_signal.v2_signal.table.soft_0.1.map@5_A` | 0.377609 | yes |
| v5 MAP@5 C | 0.3416 | `results_validation.json` | `improve_signal.v2_signal.table.soft_0.1.map@5_C` | 0.341646 | yes |
| v6 MAP@5 A | 0.3835 | `results_validation.json` | `improve_slots.v5_plus_slots.table.lambda=2.0.map@5_A` | 0.383484 | yes |
| v6 MAP@5 C | 0.3466 | `results_validation.json` | `improve_slots.v5_plus_slots.table.lambda=2.0.map@5_C` | 0.3466 | yes |
| v7 MAP@5 A | 0.3936 | `results_validation.json` | `rerank.lgbm_v6.oof.map@5_A` | 0.393562 | yes |
| v7 MAP@5 C | 0.3543 | `results_validation.json` | `rerank.lgbm_v6.oof.map@5_C` | 0.354267 | yes |
| v1 kaggle private | 0.39806 | `results_final.json` | `kaggle_submissions.submissions.0.kaggle_private` | 0.39806 | yes |
| v1 kaggle public | 0.40746 | `results_final.json` | `kaggle_submissions.submissions.0.kaggle_public` | 0.40746 | yes |
| v2 kaggle private | 0.39944 | `results_final.json` | `kaggle_submissions.submissions.1.kaggle_private` | 0.39944 | yes |
| v2 kaggle public | 0.40299 | `results_final.json` | `kaggle_submissions.submissions.1.kaggle_public` | 0.40299 | yes |
| v5 kaggle private | 0.41627 | `results_final.json` | `kaggle_submissions.submissions.2.kaggle_private` | 0.41627 | yes |
| v5 kaggle public | 0.42955 | `results_final.json` | `kaggle_submissions.submissions.2.kaggle_public` | 0.42955 | yes |
| v6 kaggle private | 0.42440 | `results_final.json` | `kaggle_submissions.submissions.3.kaggle_private` | 0.4244 | yes |
| v6 kaggle public | 0.43613 | `results_final.json` | `kaggle_submissions.submissions.3.kaggle_public` | 0.43613 | yes |
| v7 kaggle private | 0.43671 | `results_final.json` | `kaggle_submissions.submissions.4.kaggle_private` | 0.43671 | yes |
| v7 kaggle public | 0.43802 | `results_final.json` | `kaggle_submissions.submissions.4.kaggle_public` | 0.43802 | yes |

Not in a committed json: the notebook runtime (“about 1.5 minutes”) — measured by `nbconvert` in this session (92 s); per-section times are written to `outputs/nb_check/checks.json`, which is not committed. Kaggle scores in `results_final.json` were reported by the author.

