using Newtonsoft.Json;

namespace HRMS.Models.Responses
{

   public class TransferTypeResponse
   {
      [JsonProperty("Id")] public int Id { get; set; }

      [JsonProperty("TransferName")] public string TransferName { get; set; }
      [JsonProperty("BeneficiaryTypeId")] public int BeneficiaryTypeId { get; set; }

      [JsonProperty("IsEnable")] public bool IsEnable { get; set; }
   }
}