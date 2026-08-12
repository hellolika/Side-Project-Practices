CREATE PROCEDURE [dbo].[HRMS_GenerateMemberMonthlyTransfer.1.0.0]
  @startDate DATETIME,
  @endDate DATETIME
AS
BEGIN

  DECLARE @TargetMonth NVARCHAR(50)
  DECLARE @transfer TABLE (
    MemberId INT,
    TransferTypeId INT,
    DateCount INT,
    Amount DECIMAL(10,2) ,
    Remark NVARCHAR(100),
    Paydate DATETIME,
    PayStartDate DATETIME,
    PayEndDate DATETIME,
    IsGenerated BIT
    )

  SET @TargetMonth = CONVERT(NVARCHAR(10),  YEAR(@startDate)) + '-' + CONVERT(NVARCHAR(10), MONTH(@startDate))

  --handle delete member payroll from previous generate
  IF EXISTS(
    SELECT t.Id 
    FROM Transfer t 
    LEFT JOIN Member m ON t.MemberId = m.Id 
    WHERE m.[IsDeleted] = 1 AND MONTH(@startDate) = MONTH(t.PayStartDate) AND YEAR(@startDate) = YEAR(t.PayStartDate) AND MONTH(@startDate) = MONTH(t.PayEndDate) AND YEAR(@startDate) = YEAR(t.PayEndDate)
  )
  BEGIN
    DELETE t
    FROM Transfer t
    LEFT JOIN Member m ON t.MemberId = m.Id 
    WHERE m.[IsDeleted] = 1 AND MONTH(@startDate) = MONTH(t.PayStartDate) AND YEAR(@startDate) = YEAR(t.PayStartDate) AND MONTH(@startDate) = MONTH(t.PayEndDate) AND YEAR(@startDate) = YEAR(t.PayEndDate)
  END

  -- handle monthly salary
  INSERT INTO @transfer
    (MemberId,TransferTypeId,DateCount,Amount,Remark,PayDate,PayStartDate,PayEndDate,IsGenerated)
  SELECT Id , 1 AS TransferTypeId,(CASE WHEN MONTH(JoinDate) = MONTH(@startDate) AND YEAR(JoinDate) = YEAR(@startDate)
  THEN DATEDIFF(day,JoinDate,@endDate) + 1 ELSE 0 END), Salary, 'Salary of ' + @TargetMonth Remark, @startDate, @startDate, @endDate ,1
  FROM Member WITH (NOLOCK) WHERE [IsDeleted] = 0

  -- handle unpaid leave
  INSERT INTO @transfer
    (MemberId,TransferTypeId,DateCount,Amount,Remark,PayDate,[IsGenerated],PayStartDate,PayEndDate)
  SELECT a.MemberId, 2 as TransferTypeId, a.NumberOfDay, (m.Salary/30) * a.NumberOfDay AS Amount , 'UnPaid Leave of ' + @TargetMonth  AS Remark, @startDate, 1, @startDate, @endDate
  FROM (
  SELECT MemberId , LeaveType , SUM(NumberOfDay)  AS NumberOfDay
    FROM TakeLeaveRecord WITH (NOLOCK)
    where Status = 1 AND MONTH(@startDate) = MONTH(StartDate) AND  YEAR(@startDate) = YEAR(StartDate) AND MONTH(@startDate) = MONTH(EndDate) AND YEAR(@startDate) = YEAR(EndDate)
    GROUP BY MemberId , LeaveType
  ) a LEFT JOIN Member AS m WITH (NOLOCK)
    ON a.MemberId = m.Id 
  WHERE a.LeaveType = 0
  
  
  MERGE Transfer AS TARGET
    USING @transfer AS Source
    ON (TARGET.MemberId = Source.MemberId 
        AND TARGET.TransferTypeId = Source.TransferTypeId 
        AND TARGET.PayDate = Source.PayDate)
    WHEN MATCHED 
    THEN
      UPDATE 
      SET TARGET.Amount = (CASE WHEN TARGET.ModifiedBy != 0 THEN TARGET.Amount ELSE Source.Amount END),  
          TARGET.PayDate = (CASE WHEN TARGET.Status = 2 THEN TARGET.PayDate ELSE CONVERT(DATE, Source.PayStartDate) END), 
          TARGET.PayStartDate = CONVERT(DATE, Source.PayStartDate), 
          TARGET.PayEndDate = CONVERT(DATE, Source.PayEndDate)
    WHEN NOT MATCHED
    THEN 
  INSERT (MemberId, TransferTypeId, DateCount, Amount, Remark, PayDate, PayStartDate, PayEndDate, CreatedBy, CreatedOn, ModifiedBy, ModifiedOn, IsGenerated)
  VALUES (Source.MemberId, Source.TransferTypeId, Source.DateCount, Source.Amount, Source.Remark, Source.PayDate, Source.PayStartDate, Source.PayEndDate, 0, GETDATE(), 0, GETDATE(), 1);

  SELECT tf.Id as TransferId, tt.TransferName, tf.MemberId, m.Username as MemberName, tf.TransferTypeId , tf.DateCount, tf.Amount, tf.Remark, tf.PayDate
  FROM Transfer tf
  JOIN TransferType tt 
  ON tf.TransferTypeId = tt.Id
  JOIN Member m 
  ON tf.MemberId = m.Id AND m.[IsDeleted] = 0
  WHERE PayDate BETWEEN @startDate AND @endDate

END