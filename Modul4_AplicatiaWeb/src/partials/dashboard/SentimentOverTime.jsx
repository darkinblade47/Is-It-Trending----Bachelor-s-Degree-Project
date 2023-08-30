import React, { useEffect, useState } from 'react';
import LineChart from '../../charts/LineChart02';
import Info from '../../utils/Info';

import { tailwindConfig } from '../../utils/Utils';

function SentimentOverTime(props) {
  // const { dateFilter } = props;
  const apiData = props.mongoData;

  const [labels, setLabels] = useState([])
  const [datasetData, setDataset] = useState([])

  useEffect(() => {
    let reviews = []
    let filteredDataset = []
    let filteredLabels = []

    if (apiData != null) {
      reviews = apiData.review_data.filter((rev) => rev.sentiment != -1).sort((a, b) => new Date(a.date_published) - new Date(b.date_published))


      if (reviews.length > 2) {
        let startDate = new Date(reviews[0].date_published);
        let endDate = new Date(reviews[reviews.length - 1].date_published);

        if (endDate - startDate >= 345600000) {
          let countSentiment1 = 0;
          let totalCount = 0;
          const newLabels = []
          const newDataset = []
          if (reviews.length !== 0) {

            reviews.forEach((rev, index) => {
              if (rev.sentiment === 1) {
                countSentiment1 += 1;
              }
              totalCount += 1;
              newLabels.push(rev.date_published)
              newDataset.push(countSentiment1 / totalCount);
            });
          }
          filteredDataset = newDataset.filter((value, index) => index % Math.ceil(totalCount / 20) === 0)
          filteredLabels = newLabels.filter((value, index) => index % Math.ceil(totalCount / 20) === 0)

          if (filteredLabels[filteredLabels.length - 1] !== newLabels[newLabels.length - 1] &&
            filteredDataset[filteredDataset.length - 1] !== newDataset[newDataset.length - 1])
          //  bag si sentimentul curent/ultimul review din perioada de timp selectata daca nu e deja acolo
          {
            filteredLabels.push(newLabels[newLabels.length - 1])
            filteredDataset.push(newDataset[newDataset.length - 1])
          }
          for (let i = 1; i < filteredLabels.length; i++) {
            if (filteredLabels[i] == filteredLabels[i - 1]) {
              filteredLabels.splice(0, i)
              filteredDataset.splice(0, i)
            }
          }
        }
      }
      setLabels(filteredLabels);
      setDataset(filteredDataset);
    }

  }, [apiData])

  const chartData = {
    labels: labels,
    datasets: [
      // Indigo line
      {
        label: 'Grad de mulțumire',
        data: datasetData,
        borderColor: tailwindConfig().theme.colors.indigo[500],
        fill: false,
        borderWidth: 2,
        tension: 0,
        pointRadius: 3,
        pointHoverRadius: 3,
        pointBackgroundColor: tailwindConfig().theme.colors.indigo[500],
      }
    ],
  };

  return (
    <>
      <div className="flex flex-col col-span-full bg-white shadow-lg rounded-sm border border-slate-200">
        <header className="px-5 py-4 border-b border-slate-100 flex items-center">
          <h2 className="text-lg font-semibold text-slate-800">Evoluția gradului de mulțumire în timp</h2>
          <Info className="ml-2" containerClassName="min-w-80">
            <div className="text-sm">Sentimentul generat de recenziile oferite de utilizatori</div>
          </Info>
        </header>
        <LineChart data={chartData} width={595} height={248} />
      </div>
    </>
  );

}

export default SentimentOverTime;
