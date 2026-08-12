using Newtonsoft.Json;

namespace HRMS.Models.Responses
{

   public class TransferResponse
   {
      [JsonProperty("MemberId")] public int MemberId { get; set; }

      [JsonProperty("MemberName")] public string MemberName { get; set; }

      [JsonProperty("Email")] public string Email { get; set; }
      
      [JsonProperty("BankName")] public string BankName { get; set; }
      
      [JsonProperty("BankAccount")] public string BankAccount { get; set; }
      
      [JsonProperty("PhoneNumber")] public string PhoneNumber { get; set; }

      [JsonProperty("TransferName")] public string TransferName { get; set; }
      
      [JsonProperty("TeamName")] public string TeamName { get; set; }
      
      [JsonProperty("Position")] public string Position { get; set; }

      [JsonProperty("TransferId")] public int TransferId { get; set; }
      [JsonProperty("TransferTypeId")] public int TransferTypeId { get; set; }
      
      [JsonProperty("Status")] public int Status { get; set; }
      
      [JsonProperty("IsGenerated")] public bool IsGenerated { get; set; }
      
      [JsonProperty("BeneficiaryTypeId")] public int BeneficiaryTypeId { get; set; }
      [JsonProperty("DateCount")] public double DateCount { get; set; }
      
      [JsonProperty("BeneficiaryType")] public List<BeneficiaryList> BeneficiaryTypes { get; set; }

      [JsonProperty("Deduction")] public List<BeneficiaryType> Deduction { get; set; } = new List<BeneficiaryType>();

      [JsonProperty("Allowance")] public List<BeneficiaryType> Allowance { get; set; } = new List<BeneficiaryType>();

      [JsonProperty("WorkDay")] public double WorkDay { get; set; } = 0;

      [JsonProperty("Absent")] public double Absent { get; set; } = 0;
      [JsonProperty("TotalAllowanceAmount")] public double TotalAllowanceAmount { get; set; }
      [JsonProperty("TotalDeductionAmount")] public double TotalDeductionAmount { get; set; }
      
      [JsonProperty("TotalBeneficiaryAmount")] public double TotalBeneficiaryAmount { get; set; }
      [JsonProperty("NetSalary")] public double NetSalary { get; set; } = 0;

      [JsonProperty("Amount")] public double Amount { get; set; }

      [JsonProperty("Remark")] public string Remark { get; set; }
      
      [JsonProperty("ModifiedBy")] public int ModifiedBy { get; set; }
      
      [JsonProperty("Modifier")] public string Modifier { get; set; }
      
      [JsonProperty("PayDate")]
      [JsonConverter(typeof(CustomDateTimeConverter))]
      public DateTime PayDate { get; set; }
      
      [JsonProperty("PayStartDate")]
      [JsonConverter(typeof(CustomDateTimeConverter))]
      public DateTime PayStartDate { get; set; }
      
      [JsonProperty("PayEndDate")]
      [JsonConverter(typeof(CustomDateTimeConverter))]
      public DateTime PayEndDate { get; set; }
      
      [JsonProperty("CreatedOn")]
      [JsonConverter(typeof(CustomDateTimeConverter))]
      public DateTime CreatedOn { get; set; }

   }
}